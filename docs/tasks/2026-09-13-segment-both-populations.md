# タスク契約: 緑だけでなく緑＋赤の合成で細胞を分割する（対策 A）

起票: 2026-09-13 深夜、Claude Code（Fable 5.1、nd2fig のセッションから）。Shogo は就寝中で、この作業は裏で走らせる。

- **目的**(1–2行): いまの分割は Cellpose の入力が緑（Cirl-GFP）だけなので、赤（Cirl-mCherry）しか光っていない細胞は最初から
  存在しないことになる。`output/twofield_run/review/field001_compare.png` で、真っ赤な細胞に輪郭が無いことを目で確認した。
  異種の接触を数える混合指数がこれでは成り立たない。2026-08-23 に決めた対策 A（チャンネルごとに正規化してから合成した画像で分割）を実装し、
  2 視野で確かめ、通れば全視野を回す
- **完了条件**(検証可能な形で):
  1. `DirectInstanceConfig` に合成の仕方を選ぶ設定が増え、既定は今までどおり（既存の設定ファイルとテストの挙動が変わらない）。
     合成は「チャンネルごとにパーセンタイル正規化 → 最大値（または和）で 1 枚に」。Cellpose には 1 チャンネルとして渡す
  2. 合成のユニットテスト（形、チャンネルごとの正規化、赤だけの合成細胞が合成画像に現れること）と設定の検証テストがあり、
     既定の `pytest`（ML 不要の約 480 本）が全部通る
  3. 新しい設定 `configs/cirl_gfp_vs_cirl_m_both.yaml`（`input_channel_ids: [green, red]`＋合成 max）で 2 視野を通し、
     `output/twofield_both/` に出す。`review` の画像を**目で見て**、赤だけの細胞に輪郭が付き、緑の細胞が悪くなっていないことを確かめる。
     数でも比べる: 視野ごとの細胞数、`population` が Cirl-mCherry の細胞数、`field_summary.csv` の `mixing_index`／`mixing_qc`
     （field001 が `single_population` でなくなること）
  4. 3 が通ったときだけ、全視野（25）を `output/allfields_both/` に回す（1 視野 1〜2 分、全体で 1 時間前後）。通らなければ回さず、止めて報告
  5. 結果（数の比較表、時間、目で確かめたこと・数で確かめたこと、commit）をこのファイルの CHECKPOINT と「結果」節に書く
- **触ってよいファイル**(allowlist): `src/s2_adhesion/segmentation/preprocess.py`、`src/s2_adhesion/config.py`（と設定の読み込み・検証）、
  `configs/`（新しいファイルを足す。既存は変えない）、`tests/`、`docs/`。**計測（`metrics/`・`measure`）のコードは触らない**。
  `output/twofield_run/` は比較の元なので上書きしない
- **検証コマンド**:
  - テスト: `C:/Users/ryuga/dev/envs/s2-aggregate-x64/.venv/Scripts/python.exe -m pytest -q`（既定は ML を除く）
  - 実行: `C:/Users/ryuga/dev/envs/cellpose-x64/.venv/Scripts/python.exe -m s2_adhesion.cli run --config configs/cirl_gfp_vs_cirl_m_both.yaml --max-fields 2 "data/Cirl(V5)_GFP_vs_Cirl_m.nd2" output/twofield_both/`
    （cellpose 3 の venv でないと `package_major=3` の設定で落ちる）→ `... review output/twofield_both/`
- **予算の目安**: 一晩（3 時間を超えたら止めて状態を書く）。相見積もりは不要（設計は 2026-08-23 に決定済み）
- **状態**: ACTIVE（実装は済み。全視野は回していない。次は Shogo の判断待ち）

守ること:
- ブランチ `feature/segment-both-populations` を `main` から切って作業する。**`main` には触らない**（origin より 19 commit 先行して未 push のまま。push するかは Shogo の判断）。
  feature branch の push は可。PR は作らない。コミットは日本語の conventional commit（`feat:` / `fix:` / `test:` / `docs:`）
- 「テストが通った」を成果にしない。review の画像を見て、数を比べてから書く（この案件で自己申告の裏に欠陥が 3 件残った実績がある）
- 数字を文書に書くときは条件（どの設定・どの視野・何 µm 刻み）を添える
- Notion には書かない（nd2fig 側のセッションが翌朝に書く）

## CHECKPOINT(最新のみ・≤10行 — 書式: 済／次の一手／未解決・注意／検証)

- **済**: `channel_combination`（stack/max/sum、既定 stack）を実装・テスト済み（499 本通過）。`configs/cirl_gfp_vs_cirl_m_both.yaml` で 2 視野を回し `output/twofield_both/` に出した。赤だけの細胞に輪郭が付くことを目で確認
- **次の一手**: Shogo が「結果」節の比較表と未解決点を見て、赤チャンネルの正規化（上側パーセンタイル）をどうするか決める。決まったら 2 視野で再確認してから全視野
- **未解決/注意**: 赤の細胞が Z 方向に伸びすぎて（11 枚中 8〜11 枚に及ぶ）Z の端に触れ、接触の計測から外される。そのため異種接触は field000 で 2→0、field001 は `single_population` のまま。原因は赤チャンネルの 99 パーセンタイル（約 160 カウント）が細胞の明るさ（400〜700）よりずっと低く、ピンぼけの光まで白飛びするため。99.9 に上げる試行（`output/twofield_both_p999/`）は Z の伸びが減り field001 の混合指数が出るが、明るい細胞が「芯と縁」に割れる別の崩れが出る。**どちらも「明らかに合格」ではないので全視野は回していない**
- **注意 2**: 1 視野の所要時間は約 9 分（想定 1〜2 分の 5 倍）。全 25 視野なら約 4 時間
- **検証**: pytest 499 passed / 3 skipped（既定、ML 除く）。目視: `output/twofield_run` と `twofield_both` の review 画像を並べて比較。数: ラベル体積の重なりで細胞を突き合わせ（下表）

## 結果（作業した AI が書く）

作業: 2026-09-13 深夜〜早朝、Claude Code（Fable 5.1）。ブランチ `feature/segment-both-populations`（`main` から分岐、`main` は触っていない）。

### 変えたファイルと commit

| commit | 内容 |
|---|---|
| `262e1f4` feat | `src/s2_adhesion/config.py`: `DirectInstanceConfig.channel_combination`（`"stack"` / `"max"` / `"sum"`、既定 `"stack"`）と検証（不正な値、max/sum なのに 1 チャンネル、stack のときだけ cellpose 3 の 2 チャンネル上限）。`src/s2_adhesion/segmentation/preprocess.py`: `combine_channels`（チャンネルごとに正規化・リサンプル済みの配列を max または sum で 1 枚に。sum は同じ設定で再正規化）と `combined_channel_id`（`"max(green,red)"` のような id）。`direct_cellpose.py`: 診断に `input_channel_ids` と `channel_combination` を記録 |
| `80cf324` test | `tests/unit/test_preprocess.py`（合成の形・独立正規化・赤だけの合成細胞が満強度で出る・sum の再正規化、9 本）、`tests/test_config.py`（既定 stack、不正値の拒否、1 チャンネル拒否、合成なら 3 チャンネル可、新設定が読める、6 本） |
| `fb128eb` feat | `configs/cirl_gfp_vs_cirl_m_both.yaml`（既存の `cirl_gfp_vs_cirl_m.yaml` はそのまま） |
| （このあと） feat | `configs/cirl_gfp_vs_cirl_m_both_p999.yaml`（下の試行用。決定ではない） |
| （このあと） docs | この文書 |

テスト: `pytest -q`（既定、ML 除く）= **499 passed, 3 skipped, 4 deselected**（11 分 55 秒。同時に cellpose を回していたので遅い）。
既存の設定ファイルは既定 `stack` のまま読めるので、挙動は変わらない（既存テストが全部そのまま通る）。

### 2 視野の比較（数）

条件: データ `Cirl(V5)_GFP_vs_Cirl_m.nd2`、ボクセル 0.632 × 0.632 × 2.0 µm、Z 11 枚、cyto3、2.5D stitch 0.3、直径 10 µm。
「緑のみ」= `output/twofield_run/`（既存、`cirl_gfp_vs_cirl_m.yaml`）。「max p99」= `output/twofield_both/`（今回の本命、正規化 1–99%）。
「max p99.9」= `output/twofield_both_p999/`（試行。正規化の上側だけ 99.9% に）。
「消えた緑」= 緑のみの run にあった Cirl-GFP 細胞のうち、新しい run のどの細胞とも体積の 20% 未満しか重ならないもの。
「異種接触（生）」= `contacts.csv` で `qualifies_as_contact` かつ片方が Cirl-GFP・もう片方が Cirl-mCherry の数。
「異種接触（有効）」= `field_summary.csv` の `n_heterotypic_contacts`（両方の細胞が視野の端に触れていないものだけ数える）。

| 視野 | run | 細胞数 | GFP | mCherry | 曖昧 | mCherry 中央体積 µm³ | mCherry が Z 端に触れる割合 | 消えた緑 | 緑どうしの合体 | 異種接触（生） | 異種接触（有効） | mixing_index / qc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| field000 | 緑のみ | 112 | 73 | 17 | 21 | 579 | 41% | – | – | 4 | 2 | 1.07 |
| field000 | max p99 | 107 | 66 | 18 | 23 | 952 | 67% | 5（体積 528, 56, 54, 294, 83） | 1（703+887 µm³） | 4 | 0 | 0.00 |
| field000 | max p99.9 | 107 | 67 | 20 | 20 | 644 | 55% | 8 | 1（同上） | 4 | 0 | 0.00 |
| field001 | 緑のみ | 104 | 69 | 17 | 17 | 345 | 18% | – | – | 1 | 0 | single_population |
| field001 | max p99 | 105 | 68 | 21 | 15 | 553 | 48% | 4（体積 55, 54, 212, 181） | 0 | 1 | 0 | single_population |
| field001 | max p99.9 | 115 | 77 | 19 | 18 | 452 | 42% | 5 | 2（うち 1 つは 3 細胞が 1 つに） | 1 | 1 | 1.03 |

読み方:
- 赤だけの細胞は max 合成で拾えるようになった（field001 で「緑のみにはなかった新しい細胞」7 個、うち mCherry 4 個。緑のみの run で
  mCherry と付いていた 17 個は、緑の弱い漏れ込みで輪郭が付いていた細胞）
- しかし赤の細胞の体積が約 1.6 倍（579→952 µm³）に膨らみ、Z 11 枚中 8〜11 枚に及ぶものが増えた（Z 端に触れる割合 41%→67%）。
  Z 端に触れた細胞は `valid_for_geometry=False` になり接触の集計から外れるので、異種接触は「生」では 4 のまま、「有効」では 2→0 になった。
  field001 も赤を含む有効な接触が 1 つもないので `single_population` のまま
- 緑の細胞は大半がそのまま（重なり IoU の中央値 0.88〜0.91）。消えた緑は 5 個／視野で、ほぼ小さく暗いもの（体積 54〜300 µm³、
  緑の輝度が視野中央値の 1/10 前後）。1 個だけ普通の大きさ（field000 の 528 µm³、赤の細胞の隣）。緑どうしの合体は field000 に 1 組

### 目で確かめたこと

- `output/twofield_run/review/field001_compare.png` と `output/twofield_both/review/field001_compare.png` の outlines パネルを並べて見た。
  緑のみでは輪郭の無かった真っ赤な細胞 6 個（大きな 1 個を含む）に、max p99 では全部輪郭が付いた。緑の細胞の輪郭は見た目ほぼ同じ。
  新たに輪郭が付いた暗い細長い物体が 1 つある
- field000 も同様。真っ赤な細胞は元から緑の漏れ込みで輪郭が付いていたものが多く、見た目の差は小さい
- max p99.9（試行）の field001 では、明るい細胞 4〜5 個が「芯」と「縁」の二重の輪郭に割れていた（細胞数 105→115 の主因）。
  右端の細胞の塊の形も崩れている。これは目で見て分かる悪化

### 原因（数で確認）

field000 の生データで、赤チャンネルの 99 パーセンタイルは 161 カウント、緑は 1088。赤の細胞の芯は 400〜700 カウントなので、
1–99% で正規化して 1 に切り詰めると、赤の細胞はピンぼけの光（150 前後）まで真っ白になる。例: 緑のみで Z 7 枚だった赤の細胞
（field000 A19）は、赤の生データの中央値が Z ごとに 11, 40, 150, 356, 610, 418, 174, 113, 59, 27, 10 で、max p99 では Z 11 枚全部に伸びた。
緑は 99 パーセンタイルが細胞の明るさと同程度なので、この問題が起きにくい。

### 全視野の run

**回していない。** 2 視野の結果が「明らかに合格」ではないため（赤は拾えるが、Z に伸びて接触の集計から外れる）。
`output/allfields_both/` は存在しない。

### 時間

- max p99 の 2 視野: 19 分（pytest と同時に回したので遅い）。max p99.9 の 2 視野（単独）: 1120 秒 = **1 視野 約 9 分**。
  契約の見積もり（1〜2 分／視野）の約 5 倍。全 25 視野なら約 4 時間
- 作業全体: 約 3 時間（実装 40 分、テスト 12 分、run 2 回 38 分、比較と原因調べ 50 分、記録）

### Shogo に決めてほしいこと

1. 赤チャンネルの正規化をどうするか。候補: (a) チャンネルごとに別のパーセンタイルを持てるようにする（赤だけ 99.9 など。今は 1 つの
   `normalization` を両方に使う）、(b) `clip: false` で切り詰めをやめる、(c) 合成の前に赤を緑の明るさに合わせる別の方法。
   (a) は設定の形が変わるので設計の相談。試行の p99.9 は「芯と縁」に割れる副作用があったので、そのまま採用はしない方がよい
2. `mixing_qc = single_population` の意味。field001 は赤の細胞が 21 個あるのに、赤を含む「有効な」接触が 0 なので single_population と
   出る。「集団が 1 つしかない」のではなく「赤の細胞が誰とも（有効に）触れていない」。計測側の表示の話なので今回は触っていない
3. 全視野を回すなら約 4 時間かかる。夜に回すか、視野を絞るか
