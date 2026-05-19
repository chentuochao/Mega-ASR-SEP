<p align="center">
  <img src="assets/figures/mega_asr_logo.png" alt="Mega-ASR Logo" width="15%">
</p>


<h1 align="center">Mega-ASR: Towards In-the-Wild^2 Speech Recognition via Scaling Up Real-world Acoustic Simulation</h1>

We introduce **MEGA-ASR**, the first foundation ASR model to target **full-scenario robust speech recognition in the wild** through systematic training on **7 atomic acoustic conditions** and **54 compound acoustic scenarios**. Built on **2.6M training samples** covering **noise, far-field speech, obstruction, echo and reverberation, recording artifacts, electronic distortion, and transmission dropout**, MEGA-ASR uses **A2S-SFT** and **DG-WGPO based RL** to achieve **up to nearly 30% gains** over leading open and closed source SOTA models in challenging acoustic environments.

<p align="center"><u><em>You’ll come back to Mega-ASR — after finding the rest fail in the real world.</em></u></p>


<p align="center">
  <a href="https://arxiv.org/abs/2604.08000">Technical Report 📖</a>
  /
  <a href="YOUR_VOICES_IN_THE_WILD_2M_LINK">Voices-in-the-wild-2M 🤗</a>
  /
  <a href="YOUR_MEGA_ASR_WEIGHTS_LINK">Mega-ASR Weights 🤗</a>
  /
  <a href="YOUR_VOICES_IN_THE_WILD_BENCH_LINK">Voices-in-the-Wild-Bench 🏆</a>
</p>
<p align="center">
  <a href="YOUR_WECHAT_LINK_OR_QR_CODE">
    <img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white">
  </a>
  <a href="YOUR_PROJECT_PAGE_LINK">
    <img src="https://img.shields.io/badge/Project-Page-blue">
  </a>
  <a href="https://x.com/XieZhifei14110">
    <img src="https://img.shields.io/badge/X-@XieZhifei14110-black?logo=x&logoColor=white">
  </a>
</p>




<p align="center">
  <img src="/docs/assets/dataset.png" alt="Mega-ASR Logo" width="100%">
</p>

### Comparson with SOTA open-source and closed-source models.

#### Sample 1

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      ...and said to him let us go and eat some honey. Whose honey? inquired Kobay cautiously. My father's, Soongoora replied. Oh, all right, I'm with you, said the tortoise eagerly, and away they went.<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top"><span style="color:#ef4444">He</span> said to him <span style="color:#ef4444">let's</span> go and eat some honey. <span style="color:#ef4444">It's</span> honey? inquired <span style="color:#ef4444">very</span> cautiously. My father <span style="color:#ef4444">is Superabundant</span> — oh, all right, <span style="color:#ef4444">I will</span>, said <span style="color:#ef4444">to her</span> eagerly, and away they went.<br><br><strong>WER: <span style="color:#22c55e">47.1</span> ✅</strong></td><td valign="top"><span style="color:#ef4444">&lt;empty&gt;</span><br><br><strong>WER: <span style="color:#ef4444">100.0</span> 🔴</strong></td><td valign="top"><span style="color:#ef4444">But tell me, that's how she met</span> my father<span style="color:#ef4444">'s sister</span>. Oh, all right. <span style="color:#ef4444">I wish... I really...</span><br><br><strong>WER: <span style="color:#ef4444">86.1</span> 🔴</strong></td><td valign="top">My father <span style="color:#ef4444">is</span>. Oh, all right, <span style="color:#ef4444">I wish you can</span>.<br><br><strong>WER: <span style="color:#ef4444">85.3</span> 🔴</strong></td><td valign="top">...to him... some honey... <span style="color:#ef4444">oh yeah</span>...<br><br><strong>WER: <span style="color:#ef4444">92.5</span> 🔴</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

#### Sample 2

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      To waste, I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard your ship to help you in hunting the snark.<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top"><span style="color:#ef4444">To witness,</span> I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard <span style="color:#ef4444">of</span> your ship to help you in hunting the snark.<br><br><strong>WER: <span style="color:#22c55e">5.9</span> ✅</strong></td><td valign="top"><span style="color:#ef4444">I skipped 40 years. Second day in here. Ever since you left, I've been a monk...</span><br><br><strong>WER: <span style="color:#ef4444">64.7</span> 🟠</strong></td><td valign="top"><span style="color:#ef4444">I spent forty years at sea and never seen a rougher than</span> the day <span style="color:#ef4444">that</span> you took me aboard your ship...<br><br><strong>WER: <span style="color:#ef4444">64.7</span> 🟠</strong></td><td valign="top"><span style="color:#ef4444">To wait.</span> I skip forty years. <span style="color:#ef4444">Saturday and years.</span> And proceed without further remark...<br><br><strong>WER: <span style="color:#ef4444">38.2</span> 🟡</strong></td><td valign="top">I skip forty years... to the day you took me <span style="color:#ef4444">on a ship</span>... to hunt the <span style="color:#ef4444">shark</span>.<br><br><strong>WER: <span style="color:#ef4444">71.5</span> 🟠</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

<details>
<summary>More examples</summary>

#### Sample 3

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      The friendly gang left the drug store.<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top"><span style="color:#22c55e">The friendly gang left the drug store.</span><br><br><strong>WER: <span style="color:#22c55e">8.0</span> ✅</strong></td><td valign="top"><span style="color:#ef4444">It's a</span> friendly gang. <span style="color:#ef4444">That's the drug gang.</span><br><br><strong>WER: <span style="color:#ef4444">57.1</span> 🟠</strong></td><td valign="top"><span style="color:#ef4444">Friendly</span> gang left the <span style="color:#ef4444">drugs</span>.<br><br><strong>WER: <span style="color:#ef4444">42.9</span> 🟡</strong></td><td valign="top">The friendly gang left the <span style="color:#ef4444">drugstore</span>.<br><br><strong>WER: <span style="color:#22c55e">28.6</span> 🟢</strong></td><td valign="top"><span style="color:#ef4444">A</span> friendly <span style="color:#ef4444">young man</span> left the drug store.<br><br><strong>WER: <span style="color:#ef4444">62.3</span> 🟠</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

#### Sample 4

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      The set of china hit the floor with a crash.<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top"><span style="color:#22c55e">The set of china hit the floor with a crash.</span><br><br><strong>WER: <span style="color:#22c55e">8.0</span> ✅</strong></td><td valign="top">The <span style="color:#ef4444">bed is fine. It</span> hit the floor with a crash.<br><br><strong>WER: <span style="color:#ef4444">40.0</span> 🟡</strong></td><td valign="top"><span style="color:#ef4444">He said it's fine I</span> hit the <span style="color:#ef4444">forward slash</span>.<br><br><strong>WER: <span style="color:#ef4444">100.0</span> 🔴</strong></td><td valign="top">The <span style="color:#ef4444">sound</span> of china <span style="color:#ef4444">hits</span> the floor with a crash.<br><br><strong>WER: <span style="color:#22c55e">20.0</span> 🟢</strong></td><td valign="top">The <span style="color:#ef4444">chef</span> of <span style="color:#ef4444">China</span> hit the floor with a <span style="color:#ef4444">clash</span>.<br><br><strong>WER: <span style="color:#ef4444">55.0</span> 🟠</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

#### Sample 5

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      Among export-led electrical and computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty.<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top">Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty.<br><br><strong>WER: <span style="color:#22c55e">11.1</span> ✅</strong></td><td valign="top">Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan <span style="color:#ef4444">VictorNet sold fifty-two thousand three hundred fifty</span>.<br><br><strong>WER: <span style="color:#ef4444">38.9</span> 🟡</strong></td><td valign="top">Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan Victor <span style="color:#ef4444">Co.</span> fell <span style="color:#ef4444">50</span> to <span style="color:#ef4444">2,350 yen</span>.<br><br><strong>WER: <span style="color:#ef4444">35.7</span> 🟡</strong></td><td valign="top">Among export-led <span style="color:#ef4444">in</span> computer makers, Japan Victor Company <span style="color:#ef4444">sell 50 to 2300 unit</span>.<br><br><strong>WER: <span style="color:#ef4444">50.0</span> 🟠</strong></td><td valign="top">Among <span style="color:#ef4444">exporters,</span> computer makers <span style="color:#ef4444">in Japan victor companies sold</span> fifty...<br><br><strong>WER: <span style="color:#ef4444">66.7</span> 🟠</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

#### Sample 6

<table>
  <tr>
    <td valign="top">
      <video src="https://private-user-images.githubusercontent.com/201621992/594835233-2d847f22-a6d4-4d84-9bec-79a39001f9ca.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkyMDU0NDYsIm5iZiI6MTc3OTIwNTE0NiwicGF0aCI6Ii8yMDE2MjE5OTIvNTk0ODM1MjMzLTJkODQ3ZjIyLWE2ZDQtNGQ4NC05YmVjLTc5YTM5MDAxZjljYS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQxNTM5MDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mODgyYWRlZGI3OThjZWZmNzg1ZDhmNDRiNDMxZjYzZmE0Njk5OWJjYWJkZTVhZmM0OTM0OTI4MWI3ZmEzMGI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.qJS-ALDMknvRYFY73hGYmJ-WLzwtC4LRHJnHXlkpyyU" controls width="300"></video>
    </td>
    <td valign="top">
      <strong>Ground Truth</strong><br><br>
      Has exposure really been reduced?<br><br><em>Reference</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <table>
        <tr><th valign="top">Mega-ASR (Ours)</th><th valign="top">Qwen3-ASR</th><th valign="top">Gemini-3-Pro</th><th valign="top">Seed-ASR</th><th valign="top">Whisper</th></tr>
        <tr><td valign="top"><span style="color:#22c55e">Has exposure really been reduced</span><span style="color:#ef4444">.</span><br><br><strong>WER: <span style="color:#22c55e">8.0</span> ✅</strong></td><td valign="top">Has exposure really <span style="color:#ef4444">done you?</span><br><br><strong>WER: <span style="color:#ef4444">40.0</span> 🟡</strong></td><td valign="top">Has <span style="color:#ef4444">the closure</span> really <span style="color:#ef4444">affected you?</span><br><br><strong>WER: <span style="color:#ef4444">80.0</span> 🔴</strong></td><td valign="top">Has exposure <span style="color:#ef4444">to beauty products.</span><br><br><strong>WER: <span style="color:#ef4444">60.0</span> 🟠</strong></td><td valign="top"><span style="color:#ef4444">Have those who</span> really <span style="color:#ef4444">been refused?</span><br><br><strong>WER: <span style="color:#ef4444">78.5</span> 🔴</strong></td></tr>
      </table>
    </td>
  </tr>
</table>

</details>

<!-- 
### Comparson with SOTA open-source and closed-source models.

| Audio | Ground Truth | Mega-ASR (Ours) | Qwen3-ASR | Gemini-3-Pro | Seed-ASR | Whisper |
|---|---|---|---|---|---|---|
| <video src="assets/case_study/empty_output_recovery.mp4" controls width="240"></video> | ...and said to him let us go and eat some honey. Whose honey? inquired Kobay cautiously. My father's, Soongoora replied. Oh, all right, I'm with you, said the tortoise eagerly, and away they went.<br><br>*Reference* | <span style="color:#ef4444">He</span> said to him <span style="color:#ef4444">let's</span> go and eat some honey. <span style="color:#ef4444">It's</span> honey? inquired <span style="color:#ef4444">very</span> cautiously. My father <span style="color:#ef4444">is Superabundant</span> — oh, all right, <span style="color:#ef4444">I will</span>, said <span style="color:#ef4444">to her</span> eagerly, and away they went.<br><br>**WER: <span style="color:#22c55e">47.1</span> ✅** | <span style="color:#ef4444">&lt;empty&gt;</span><br><br>**WER: <span style="color:#ef4444">100.0</span> 🔴** | <span style="color:#ef4444">But tell me, that's how she met</span> my father<span style="color:#ef4444">'s sister</span>. Oh, all right. <span style="color:#ef4444">I wish... I really...</span><br><br>**WER: <span style="color:#ef4444">86.1</span> 🔴** | My father <span style="color:#ef4444">is</span>. Oh, all right, <span style="color:#ef4444">I wish you can</span>.<br><br>**WER: <span style="color:#ef4444">85.3</span> 🔴** | ...to him... some honey... <span style="color:#ef4444">oh yeah</span>...<br><br>**WER: <span style="color:#ef4444">92.5</span> 🔴** |
| <video src="assets/case_study/long_utterance_recovery.mp4" controls width="240"></video> | To waste, I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard your ship to help you in hunting the snark.<br><br>*Reference* | <span style="color:#ef4444">To witness,</span> I skip forty years, said the baker in tears, and proceed without further remark to the day when you took me aboard <span style="color:#ef4444">of</span> your ship to help you in hunting the snark.<br><br>**WER: <span style="color:#22c55e">5.9</span> ✅** | <span style="color:#ef4444">I skipped 40 years. Second day in here. Ever since you left, I've been a monk...</span><br><br>**WER: <span style="color:#ef4444">64.7</span> 🟠** | <span style="color:#ef4444">I spent forty years at sea and never seen a rougher than</span> the day <span style="color:#ef4444">that</span> you took me aboard your ship...<br><br>**WER: <span style="color:#ef4444">64.7</span> 🟠** | <span style="color:#ef4444">To wait.</span> I skip forty years. <span style="color:#ef4444">Saturday and years.</span> And proceed without further remark...<br><br>**WER: <span style="color:#ef4444">38.2</span> 🟡** | I skip forty years... to the day you took me <span style="color:#ef4444">on a ship</span>... to hunt the <span style="color:#ef4444">shark</span>.<br><br>**WER: <span style="color:#ef4444">71.5</span> 🟠** |

<details>
<summary>More examples</summary>

| Audio | Ground Truth | Mega-ASR (Ours) | Qwen3-ASR | Gemini-3-Pro | Seed-ASR | Whisper |
|---|---|---|---|---|---|---|
| <video src="assets/case_study/babble_noise_hallucination.mp4" controls width="240"></video> | The friendly gang left the drug store.<br><br>*Reference* | <span style="color:#22c55e">The friendly gang left the drug store.</span><br><br>**WER: <span style="color:#22c55e">8.0</span> ✅** | <span style="color:#ef4444">It's a</span> friendly gang. <span style="color:#ef4444">That's the drug gang.</span><br><br>**WER: <span style="color:#ef4444">57.1</span> 🟠** | <span style="color:#ef4444">Friendly</span> gang left the <span style="color:#ef4444">drugs</span>.<br><br>**WER: <span style="color:#ef4444">42.9</span> 🟡** | The friendly gang left the <span style="color:#ef4444">drugstore</span>.<br><br>**WER: <span style="color:#22c55e">28.6</span> 🟢** | <span style="color:#ef4444">A</span> friendly <span style="color:#ef4444">young man</span> left the drug store.<br><br>**WER: <span style="color:#ef4444">62.3</span> 🟠** |
| <video src="assets/case_study/restaurant_noise_recovery.mp4" controls width="240"></video> | The set of china hit the floor with a crash.<br><br>*Reference* | <span style="color:#22c55e">The set of china hit the floor with a crash.</span><br><br>**WER: <span style="color:#22c55e">8.0</span> ✅** | The <span style="color:#ef4444">bed is fine. It</span> hit the floor with a crash.<br><br>**WER: <span style="color:#ef4444">40.0</span> 🟡** | <span style="color:#ef4444">He said it's fine I</span> hit the <span style="color:#ef4444">forward slash</span>.<br><br>**WER: <span style="color:#ef4444">100.0</span> 🔴** | The <span style="color:#ef4444">sound</span> of china <span style="color:#ef4444">hits</span> the floor with a crash.<br><br>**WER: <span style="color:#22c55e">20.0</span> 🟢** | The <span style="color:#ef4444">chef</span> of <span style="color:#ef4444">China</span> hit the floor with a <span style="color:#ef4444">clash</span>.<br><br>**WER: <span style="color:#ef4444">55.0</span> 🟠** |
| <video src="assets/case_study/financial_entity_recovery.mp4" controls width="240"></video> | Among export-led electrical and computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty.<br><br>*Reference* | Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan Victor Company fell fifty to two thousand three hundred twenty.<br><br>**WER: <span style="color:#22c55e">11.1</span> ✅** | Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan <span style="color:#ef4444">VictorNet sold fifty-two thousand three hundred fifty</span>.<br><br>**WER: <span style="color:#ef4444">38.9</span> 🟡** | Among export-led <span style="color:#ef4444">(missing: electrical and)</span> computer makers, Japan Victor <span style="color:#ef4444">Co.</span> fell <span style="color:#ef4444">50</span> to <span style="color:#ef4444">2,350 yen</span>.<br><br>**WER: <span style="color:#ef4444">35.7</span> 🟡** | Among export-led <span style="color:#ef4444">in</span> computer makers, Japan Victor Company <span style="color:#ef4444">sell 50 to 2300 unit</span>.<br><br>**WER: <span style="color:#ef4444">50.0</span> 🟠** | Among <span style="color:#ef4444">exporters,</span> computer makers <span style="color:#ef4444">in Japan victor companies sold</span> fifty...<br><br>**WER: <span style="color:#ef4444">66.7</span> 🟠** |
| <video src="assets/case_study/phrase_recovery.mp4" controls width="240"></video> | Has exposure really been reduced?<br><br>*Reference* | <span style="color:#22c55e">Has exposure really been reduced</span><span style="color:#ef4444">.</span><br><br>**WER: <span style="color:#22c55e">8.0</span> ✅** | Has exposure really <span style="color:#ef4444">done you?</span><br><br>**WER: <span style="color:#ef4444">40.0</span> 🟡** | Has <span style="color:#ef4444">the closure</span> really <span style="color:#ef4444">affected you?</span><br><br>**WER: <span style="color:#ef4444">80.0</span> 🔴** | Has exposure <span style="color:#ef4444">to beauty products.</span><br><br>**WER: <span style="color:#ef4444">60.0</span> 🟠** | <span style="color:#ef4444">Have those who</span> really <span style="color:#ef4444">been refused?</span><br><br>**WER: <span style="color:#ef4444">78.5</span> 🔴** |

</details> -->



## 🔥News

- **May 21, 2025**: 🔥 We release **Voices-in-the-Wild-Bench**, a benchmark for in-the-wild ASR robustness evaluation.
- **May 20, 2025**: 🔥 We release **Voices-in-the-Wild-2M**.
- **May 20, 2025**: 🔥 We release the **Mega-ASR Inference and Training Codebase**.
- **May 19, 2025**: 🔥 **Mega-ASR** model weights are now available on Hugging Face.
- **May 19, 2025**: 🔥 We release the **Mega-ASR Technical Report**.

## Overview

* **[Quick Start](#quick-start)**
* **[Introduction](#inference)**
* **[Inference and deployment](#inference)**
* **[Finetuning](#finetune)**
* **[Evaluation](#evaluation)**
* **[Citation and licence](#citation)**


## Quick Start

**Installation**
```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
conda create -n mega-asr2 python=3.12 -y
conda activate mega-asr2

pip install torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0
pip install -r mega_asr_requirements.txt
pip install -e /path/to/Qwen3-ASR --no-deps
```
**Download**
```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
```

**Offline Inference**
```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
```

**WEBUI**
```bash
git clone https://github.com/QwenLM/Qwen3-ASR.git
```


## Introduction

**MEGA-ASR** is purpose-built for **full-scenario robust ASR in the wild**, especially excelling at **semantic recovery** and **local keyword reconstruction** under severe acoustic degradation. It substantially reduces common failure modes such as **hallucinations**, **empty outputs**, and **dropped utterances**, making speech recognition reliable in truly challenging real-world environments.
<p align="center">
  <img src="assets/figures/radar_results.png" alt="Results" width="100%">
</p>

#### Features 
✅ **One model for the messy real world**: Covers **7 atomic acoustic conditions** and **54 compound acoustic scenarios** in a single model.

✅ **Stronger recovery under severe distortion**: Excels at **semantic recovery** and **local keyword reconstruction**, greatly reducing **hallucinations**, **empty outputs**, and **dropped utterances**.

✅ **SOTA robust ASR performance**: Achieves up to nearly **30% gains** over leading open and closed source SOTA models in challenging acoustic environments.





## Finetuning

Mega-ASR supports robustness adaptation through supervised fine-tuning (A2S-SFT) and reinforcement learning (DG-WGPO).

### A2S-SFT


`src/MegaASR/A2S-SFT` contains the core training code for Mega-ASR A2S-SFT. 

```text
src/MegaASR/A2S-SFT/
├── arguments.py      # Defines command-line arguments and training hyperparameters.
├── checkpointing.py  # Saves base-model metadata and required processor/tokenizer files for LoRA reuse.
├── dataloader.py     # Loads JSONL data, reads audio, builds model inputs, and masks non-target tokens.
├── finetune.py       # Main entry point for launching A2S-SFT training.
├── modeling.py       # Loads Qwen3-ASR and defines LoRA injection scopes.
├── trainer.py        # Defines MegaASRTrainer with adapter-only saving and module-wise learning rates.
```


Training data is in JSONL format:

```json
{
  "audio": ".../wavs/test-clean/61/70968/61-70968-0000.wav",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": ""
}
```

We can use the following command to start it.

```bash
torchrun --nproc_per_node=2 A2S_SFT/finetune.py \
  --model_path Qwen3-ASR-1.7B --train_file ${TRAIN_JSONL} \
  --eval_file ${VAL_JSONL} --output_dir ${OUT_DIR} \
  --batch_size 8 --grad_acc 8 \
  --lr 1e-6 --lr_encoder 1e-6 --lr_aligner 1e-6 --lr_llm 1e-6 \
  --epochs 2 --save_steps 200 --save_total_limit 300 --use_lora 1 \
  --lora_scope all --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --warmup_ratio 0.05 --max_grad_norm 1.0  --weight_decay 0.01 \
  --run_name ${RUN_NAME} --report_to wandb \
  2>&1 | tee -a ${LOG_FILE}
```

The DG-WGPO reinforcement learning module will be released in a future update.

## Evaluation


We provide a simple evaluation script for running ASR inference and computing WER/CER.  We use Qwen3-ASR as the default inference model. The input file should be a JSONL file. Each line only needs two required fields:

```json
{"audio": "examples/audio/noise.wav", "answer": "I usually take the quieter road home because the main street gets crowded after work."}
```


The script will keep all original fields and append the following fields to the output JSONL:

```text
prediction  # model transcription
metric      # "wer" for English samples, "cer" for Chinese samples
wer         # WER/CER score value; CER is also stored in this field for compatibility
num_edits   # edit distance between prediction and ground truth
ref_len     # number of reference words or characters
```
We can use the following command to start it.

```bash
python eval/evaluate_asr.py \
  --model_path Qwen3-ASR-1.7B \
  --input_jsonl examples/test.jsonl \
  --output_jsonl outputs/pred_with_wer.jsonl
```

<p align="center">
  <img src="/docs/assets/training.png" alt="Mega-ASR Training" width="100%">
</p>

**Mega-ASR** is trained with an acoustic-to-semantic progressive supervised fine-tuning strategy: it first curriculum-trains the encoder and aligner on increasingly difficult samples from WER<30% to WER<50% and then WER<70%, then fine-tunes the LLM on WER<70% data to strengthen semantic recovery, and finally jointly fine-tunes the full encoder-aligner-LLM stack for end-to-end alignment.

On top of Mega-ASR-Base, DG-WGPO further optimizes the model with WER-gated policy learning: low-WER samples emphasize token-level acoustic refinement, while high-WER samples emphasize sentence-level semantic reconstruction to reduce hallucinations, omissions, and off-audio outputs. The final reward combines a static WER-based accuracy signal with an anti-repetition gate and a dynamic dual-granularity reward, using fixed hyperparameters τ=0.3, αs=0.4, and αdyn=0.6.


Run Qwen3-ASR inference and compute WER (English) / CER (Chinese) on JSONL data:

```bash
CUDA_VISIBLE_DEVICES=6,7 python evaluate_wer.py \
  --input_jsonl example/examples.jsonl \
  --output_jsonl output_with_wer.jsonl
```

Each input line requires `audio_path` and `answer` (ground-truth transcription). Place `evaluate_wer.py` and `cn_tn.py` (used for Chinese text normalization) in the same directory.

**Mega-ASR** is evaluated across three benchmark families — classical academic test sets, robustness benchmarks, and our own in-the-wild compound benchmark.


<p align="center">
  <img src="/assets/tables/noisy_robust_asr_benchmarks.png" alt="Mega-ASR Results" width="100%">
</p>

<p align="center">
  <img src="/assets/tables/voices_in_the_wild_breakdown.png" alt="Mega-ASR Results" width="100%">
</p>


## Acknowledgements


## License
This project will be released under the Apache-2.0 License.


## Citation and stars
```bash
@article{xie2024mini,
  title={Mini-omni: Language models can hear, talk while thinking in streaming},
  author={Xie, Zhifei and Wu, Changqiao},
  journal={arXiv preprint arXiv:2408.16725},
  year={2024}
}
```

<a href="https://www.star-history.com/?repos=mega-asr%2Fmega-asr&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=mega-asr/mega-asr&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=mega-asr/mega-asr&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=mega-asr/mega-asr&type=date&legend=top-left" />
 </picture>
</a>

