# coding=utf-8
import math
import random

from torch.utils.data import DataLoader, Sampler
from transformers import TrainerCallback


class SamplerEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if train_dataloader is None:
            return control

        batch_sampler = getattr(train_dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            epoch = int(state.epoch) if state.epoch is not None else 0
            batch_sampler.set_epoch(epoch)
        return control


class FixedRatioAccumBatchSampler(Sampler):
    """
    语义：
    - 控制“全局一个 optimizer step”内的 targeted/generic 比例
    - 全局 effective batch = micro_batch_size * grad_acc * world_size
    - 每个 rank 的 __len__ 完全一致，适合 DDP
    - targeted 不够时允许 oversample
    - generic 作为 epoch 长度基准（默认 drop_last）
    """

    def __init__(
        self,
        targeted_indices,
        generic_indices,
        micro_batch_size: int,
        grad_acc: int,
        target_ratio: float = 0.3,
        seed: int = 42,
        process_index: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        self.seed = seed
        self.epoch = 0
        self.micro_batch_size = micro_batch_size
        self.grad_acc = grad_acc
        self.target_ratio = target_ratio
        self.drop_last = drop_last
        self.process_index = process_index
        self.world_size = world_size

        # 注意：这里保留“全量全局索引”，不再先按 rank 切片
        self.targeted_indices = list(targeted_indices)
        self.generic_indices = list(generic_indices)

        if len(self.targeted_indices) == 0:
            raise ValueError("No targeted samples found in the whole dataset.")
        if len(self.generic_indices) == 0:
            raise ValueError("No generic samples found in the whole dataset.")

        # local = 单卡一个 optimizer step 的样本数
        self.local_effective_batch_size = self.micro_batch_size * self.grad_acc

        # global = 所有卡合起来一个 optimizer step 的样本数
        self.global_effective_batch_size = (
            self.local_effective_batch_size * self.world_size
        )

        # 全局一个 optimizer step 内精确控制比例
        self.target_per_global_update = round(
            self.global_effective_batch_size * self.target_ratio
        )
        self.generic_per_global_update = (
            self.global_effective_batch_size - self.target_per_global_update
        )

        if self.target_per_global_update <= 0 or self.generic_per_global_update <= 0:
            raise ValueError(
                f"Bad ratio setting: "
                f"target_per_global_update={self.target_per_global_update}, "
                f"generic_per_global_update={self.generic_per_global_update}"
            )

        # epoch 长度按“全局 generic 池”来定，保证各 rank 一致
        if drop_last:
            self.num_updates = len(self.generic_indices) // self.generic_per_global_update
        else:
            self.num_updates = math.ceil(
                len(self.generic_indices) / self.generic_per_global_update
            )

        if self.num_updates <= 0:
            raise ValueError(
                "num_updates <= 0, generic dataset too small for one global update."
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        # 每个 rank 在一个 epoch 内都会 yield 同样多的 micro-batches
        return self.num_updates * self.grad_acc

    @staticmethod
    def _split_integer_across_parts(total: int, parts: int, rng: random.Random):
        """
        把 total 尽量均匀分到 parts 份，和为 total。
        多出来的 remainder 随机分配到若干份上。
        """
        base = total // parts
        rem = total % parts
        out = [base] * parts
        if rem > 0:
            extra_pos = rng.sample(range(parts), rem)
            for pos in extra_pos:
                out[pos] += 1
        return out

    def __iter__(self):
        # 注意：所有 rank 用同一个 epoch 级随机种子，确保“全局计划”一致
        rng = random.Random(self.seed + self.epoch * 100003)

        t_pool = list(self.targeted_indices)
        g_pool = list(self.generic_indices)
        rng.shuffle(t_pool)
        rng.shuffle(g_pool)

        t_ptr = 0
        g_ptr = 0

        for update_idx in range(self.num_updates):
            # 1) 先取出“全局一个 optimizer step”需要的 target/generic
            need_t = self.target_per_global_update
            need_g = self.generic_per_global_update

            # targeted 不够就 oversample
            t_chunk = []
            while len(t_chunk) < need_t:
                remain = len(t_pool) - t_ptr
                if remain <= 0:
                    rng.shuffle(t_pool)
                    t_ptr = 0
                    remain = len(t_pool)

                take = min(need_t - len(t_chunk), remain)
                t_chunk.extend(t_pool[t_ptr:t_ptr + take])
                t_ptr += take

            # generic 默认不 oversample（drop_last=True 时本来就够）
            if g_ptr + need_g > len(g_pool):
                if self.drop_last:
                    return
                g_chunk = []
                while len(g_chunk) < need_g:
                    remain = len(g_pool) - g_ptr
                    if remain <= 0:
                        rng.shuffle(g_pool)
                        g_ptr = 0
                        remain = len(g_pool)
                    take = min(need_g - len(g_chunk), remain)
                    g_chunk.extend(g_pool[g_ptr:g_ptr + take])
                    g_ptr += take
            else:
                g_chunk = g_pool[g_ptr:g_ptr + need_g]
                g_ptr += need_g

            # 2) 把全局 target 总数分配到 grad_acc 个 micro-step
            global_t_per_micro = self._split_integer_across_parts(
                self.target_per_global_update,
                self.grad_acc,
                rng,
            )

            # 指针：在这个 global update 的 target/generic chunk 内切片
            t_off = 0
            g_off = 0

            for micro_idx, t_global_this_micro in enumerate(global_t_per_micro):
                # 当前 micro-step 的全局 generic 数
                g_global_this_micro = (
                    self.micro_batch_size * self.world_size - t_global_this_micro
                )

                # 3) 先只分 target；generic 由每个 rank 自动补齐到 micro_batch_size
                t_per_rank = self._split_integer_across_parts(
                    t_global_this_micro,
                    self.world_size,
                    rng,
                )
                g_per_rank = [self.micro_batch_size - tr for tr in t_per_rank]

                # 安全检查
                if sum(g_per_rank) != g_global_this_micro:
                    raise RuntimeError(
                        f"Generic split mismatch: sum(g_per_rank)={sum(g_per_rank)} "
                        f"!= g_global_this_micro={g_global_this_micro}"
                    )

                for tr, gr in zip(t_per_rank, g_per_rank):
                    if tr < 0 or gr < 0 or tr + gr != self.micro_batch_size:
                        raise RuntimeError(
                            f"Per-rank micro batch mismatch: {tr} + {gr} != {self.micro_batch_size}"
                        )

                # 4) 所有 rank 都按同一个“全局计划”切自己的那份
                for rank in range(self.world_size):
                    t_cnt = t_per_rank[rank]
                    g_cnt = g_per_rank[rank]

                    batch = (
                        t_chunk[t_off:t_off + t_cnt] +
                        g_chunk[g_off:g_off + g_cnt]
                    )
                    t_off += t_cnt
                    g_off += g_cnt

                    # 每个 local batch 内再打散一下
                    batch_rng = random.Random(
                        self.seed
                        + self.epoch * 100003
                        + update_idx * 997
                        + micro_idx * 53
                        + rank
                    )
                    batch_rng.shuffle(batch)

                    if rank == self.process_index:
                        yield batch


class FixedRatioTrainLoaderMixin:
    def __init__(
        self,
        *args,
        mix_target_ratio=0.2,
        mix_domain_field="domain",
        mix_target_value="targeted",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mix_target_ratio = mix_target_ratio
        self.mix_domain_field = mix_domain_field
        self.mix_target_value = mix_target_value

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        domains = self.train_dataset[self.mix_domain_field]

        targeted_indices = [
            i for i, d in enumerate(domains)
            if d == self.mix_target_value
        ]
        generic_indices = [
            i for i, d in enumerate(domains)
            if d != self.mix_target_value
        ]

        if len(targeted_indices) == 0:
            raise ValueError(
                f"No targeted samples found where "
                f"{self.mix_domain_field}={self.mix_target_value}"
            )
        if len(generic_indices) == 0:
            raise ValueError("No generic samples found.")

        batch_sampler = FixedRatioAccumBatchSampler(
            targeted_indices=targeted_indices,
            generic_indices=generic_indices,
            micro_batch_size=self.args.per_device_train_batch_size,
            grad_acc=self.args.gradient_accumulation_steps,
            target_ratio=self.mix_target_ratio,
            seed=self.args.seed,
            process_index=self.args.process_index,
            world_size=self.args.world_size,
            drop_last=True,
        )

        if self.args.process_index == 0:
            global_bs = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.args.world_size
            )
            target_num = round(global_bs * self.mix_target_ratio)
            generic_num = global_bs - target_num
            print(
                f"[sampler] global_effective_batch={global_bs}, "
                f"targeted={target_num}, generic={generic_num}, "
                f"ratio={self.mix_target_ratio}"
            )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=(
                self.args.dataloader_prefetch_factor
                if self.args.dataloader_num_workers > 0
                else None
            ),
        )
