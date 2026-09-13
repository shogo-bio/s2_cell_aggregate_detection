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
- **状態**: ACTIVE（赤の正規化は決定・実装済み。2 視野合格。全 25 視野を実行中 — 2026-09-13 13:47 JST 開始）

## 下流が待っている（2026-09-13 追記）

この定量の結果は、図を作る道具 **nd2fig**（`C:\Users\ryuga\dev\nd2fig`）が読む。流れは「① 定量（このリポジトリ）→ ② 定量結果から
切り抜く場所の候補を出す（nd2fig の Phase 7、契約 `docs/tasks/quant-to-figure-bridge.md`）→ ③ 図にする（nd2fig、完成済み）」。
nd2fig 側は①が固まるまで**待機中**。①に頼むことは 2 つ:

1. 分割が合格したら、**全 25 視野**の `objects.csv`・`aggregates.csv`・`field_summary.csv`・`run_manifest.json` を 1 か所に出す
   （出力先の名前を CHECKPOINT に書く。1 視野 9 分なので約 4 時間。電源をつないで回す）
2. nd2fig が当てにしている列を**変えない**（変えるなら CHECKPOINT に書く）: `objects.csv` の `field_id`・`centroid_x_um / centroid_y_um /
   centroid_z_um`・`bbox_extent_{x,y,z}_um`・`aggregate_id`・`touches_z_border`・`population`・`ch.*.saturation_status`、
   `aggregates.csv` の `member_cell_ids`・`aggregate_extent_{x,y,z}_um`・`aggregate_cell_count`、`field_summary.csv` の `mixing_index`・`mixing_qc`。
   座標はいまの決まり（ボクセル中心、配列の原点 = 視野の左上、µm）のまま

守ること:
- ブランチ `feature/segment-both-populations` を `main` から切って作業する。**`main` には触らない**（origin より 19 commit 先行して未 push のまま。push するかは Shogo の判断）。
  feature branch の push は可。PR は作らない。コミットは日本語の conventional commit（`feat:` / `fix:` / `test:` / `docs:`）
- 「テストが通った」を成果にしない。review の画像を見て、数を比べてから書く（この案件で自己申告の裏に欠陥が 3 件残った実績がある）
- 数字を文書に書くときは条件（どの設定・どの視野・何 µm 刻み）を添える
- Notion には書かない（nd2fig 側のセッションが翌朝に書く）

## CHECKPOINT(最新のみ・≤10行 — 書式: 済／次の一手／未解決・注意／検証)

- **済**: 赤の正規化を Codex（GPT-6）と Fable に相見積もり → 両者一致で「赤の上側を絶対値 1000 カウントで固定」（下の「結果 2」）。`normalization_by_channel` と `upper_value` を実装（commit `8eed11d`、pytest 522 passed / 3 skipped）。`configs/cirl_gfp_vs_cirl_m_both.yaml` を赤 `upper_value: 1000` に更新し、2 視野（`output/twofield_both_abs1000/`）と field020（`output/onefield020_abs1000/`）で合格を確認
- **次の一手**: **全 25 視野を `output/allfields_both/` に向けて実行中**（13:47 JST 開始、1 視野 約 6〜9 分）。終わったら review 画像を数視野見て、視野ごとの表をこの文書に書き、commit する。nd2fig は `output/allfields_both/measurements/`（`objects.csv`・`aggregates.csv`・`contacts.csv`・`field_summary.csv`）と `output/allfields_both/run_manifest.json` を読む
- **未解決/注意**: (1) 暗い赤だけの細胞（赤の中央値 50〜60 カウント）は拾わない（設計上の優先順位: Z 伸びの回避 > 暗い赤の検出。1–99 の設定では拾えたが Z に伸びた）。(2) field001 は正しい分割でも `single_population`（異種接触が生で 1 つだけで、その緑側が Z 端）。契約の完了条件 3 の「field001 が single_population でなくなる」は**達成していない**が、相見積もり両者とも「接触の有無を合否条件にしない」（正しい分割でも 0 になり得る）。(3) `field_summary.csv` の `contacts.<集団>__<集団>` 列は存在する接触の組み合わせだけ出る（`contacts.Cirl-mCherry__Cirl-mCherry` が新たに出る視野がある）。nd2fig が当てにしている列（`mixing_index`・`mixing_qc` ほか）は変えていない
- **検証**: pytest 522 passed / 3 skipped / 4 deselected（既定、ML 除く、7 分 49 秒）。目視: `twofield_both_abs1000/review/field00{0,1}_compare.png` と `onefield020_abs1000/review/field020_compare.png`。数: 下の「結果 2」の表（緑のみ・max p99・max 赤 1000 の 3 者比較と、細胞ごとの前後対応）

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

## 結果 2（2026-09-13 午後: 赤の正規化の決定 → 2 視野の再確認 → 全視野）

作業: 2026-09-13 13:05〜、Claude Code（Fable 5.1）。Shogo の指示「赤の正規化を Codex に相見積もりしてから 2 視野で試し、合格したら全 25 視野を回す。main は触らない」。

### 相見積もり（Codex = GPT-6 と名乗り、`gpt-6-astra` 指定 / Fable = Fable 5.1 と名乗り）

ブリーフは scratchpad の `consult_red_normalization.md`（背景・2 視野の実測・25 視野の赤の分布・候補 A〜E・私の案）。私の案は **A「赤だけ上側 99.9 パーセンタイル」**を本命、C「赤の上側を絶対カウントで固定」を控えにしていた。

| 争点 | Codex | Fable | 一致/相違 |
|---|---|---|---|
| 赤の正規化 | **C（赤 1000 カウント固定）を本命**、A は比較対象。B（clip なし）は最初の 2 案から外す。D（下限付き p99.9）は根拠が薄いので見送り | **A 単体に反対、C か D を本命**。B は「Z 伸びの原因は切り詰めではなく 1/161 のスケール」なので却下 | **一致: C。私の A は両者から否定** |
| A への反論 | field020 では A が C の 2.26 倍に赤を増幅し、ピンぼけ 150 カウントが 0.34 になって緑の芯（0.2〜0.5）と同じ帯に入る | p99.9 は「最も明るい約 4.4 個分の細胞ボクセルの下端」＝**個数の統計**。赤が少ない視野で p99.9 がピンぼけまで落ち、赤ゼロの視野（単一集団コントロール）では緑まで壊す | 一致 |
| 設定の形 | `normalization_by_channel` で「上限の決め方の置き換え」と定義。同じブロックに `upper_percentile` と `upper_value` の両方があればエラー | 同じ。`_coerce` が `Mapping[str, dataclass]` を扱えないので tuple 形（`[{channel_id: red, upper_value: 1000}]`）を推奨。sum との併用はエラーに | 一致（tuple 形で実装） |
| 2 視野の試行 | field000 と field020（赤が疎）に A/C の計 4 本。**「有効な異種接触が 0 でなくなる」を必須条件にすることに反対**（正しい分割でも 0 になり得る、過小分割で人工的に増やせる） | C を field000/001、A と C を field020。合否は Z 端率でなく「Z 広がりの中央値・≥9 面の割合・体積中央値・個数」 | 相違は視野の組み方だけ |
| 落とし穴 | max 合成では赤が緑を隠す画素がある。Z 端率は赤を失っても改善する（検出率と一対一対応を一緒に見る）。固定カウントは背景・深さ・褪色で崩れ得る | 赤ゼロの視野、暗い赤の系統的欠落（混合指数を高発現側に寄せる）、合成画像の p99 が 1 から外れる条件 | 補完的 |

私の判断: **C を採用**。決め手は相見積もりに加えて次の診断（分割不要、ND2 から直接計算、Fable の提案）。

### 診断: 赤の p99.9 は「個数」か「明るさ」か（25 視野、`Cirl(V5)_GFP_vs_Cirl_m.nd2`）

| 相関（25 視野） | r |
|---|---|
| 赤 p99.9 と 赤の塊の個数（≥300 カウントの 3D 連結成分、≥100 ボクセル） | **0.88** |
| 赤 p99.9 と 明るい赤の芯の明るさ（各塊の p90 の中央値） | **0.19** |
| 赤 p99.9 と 緑 p99 | 0.61（細胞密度が両方に効いている。赤の個数と緑の個数の相関 0.79） |

明るい赤の芯の明るさ自体は 25 視野で 726〜1191 カウント（中央値約 950）と安定。赤 p99.9 は 442〜1233（2.8 倍）で個数に追従する。→ 同じ ND2（同じ検出器設定）の中では絶対値 1000 が安定。

### 実装（commit `8eed11d`）

- `NormalizationConfig.upper_value`（絶対カウントの上側。指定時は `upper_percentile` を無視）
- `ChannelNormalizationOverride` と `CellposeModelConfig.normalization_by_channel`（書いた項目だけ共通設定を置き換える。`resolve()` で解決）
- 検証: 未知のチャンネル id・同じチャンネルの二重指定・`upper_percentile` と `upper_value` の同時指定・`sum` との併用・direct_cellpose 以外（watershed / StarDist）での使用を `ConfigError`
- `preprocess.prepare_cellpose_input` はチャンネルごとに解決した設定で正規化し、実効 lo/hi を `logging` に出す（`normalization_bounds`）
- `configs/cirl_gfp_vs_cirl_m_both.yaml`: 赤だけ `upper_value: 1000`。緑は 1–99 のまま。既存の設定ファイルの挙動は不変
- テスト 23 本追加。pytest 522 passed / 3 skipped / 4 deselected（既定）

### 2 視野の比較（数）

条件: 上と同じ（ボクセル 0.632 × 0.632 × 2.0 µm、Z 11 枚、cyto3、2.5D stitch 0.3、直径 10 µm）。「max 赤1000」= `output/twofield_both_abs1000/`。
体積は `volume_um3`（前回の表の `579` などは別の体積列。行どうしの比較は同じ列で）。「Z 端率」= `touches_z_border` の割合。「消えた緑」「緑の合体」は前回と同じ定義（体積の重なり 20% / 50%）。

| 視野 | run | 細胞数 | GFP | mCherry | 曖昧 | mCherry 体積中央値 µm³ | mCherry Z 範囲中央値 µm | mCherry Z 端率 | GFP Z 端率 | 消えた緑 | 緑の合体 | 異種接触（生） | 異種接触（有効） | mixing_index / qc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| field000 | 緑のみ | 112 | 73 | 17 | 21 | 383 | 9 | 41% | 16% | – | – | 4 | 2 | 1.07 |
| field000 | max p99 | 107 | 66 | 18 | 23 | 454 | 9 | 67% | 23% | 5 | 2 | 4 | 0 | 0.00 |
| field000 | **max 赤1000** | 113 | 72 | 18 | 21 | 376 | 8 | 39% | 17% | **0** | 1 | 4 | **2** | 0.57 |
| field001 | 緑のみ | 104 | 69 | 17 | 17 | 292 | 10 | 18% | 35% | – | – | 1 | 0 | single_population |
| field001 | max p99 | 105 | 68 | 21 | 15 | 163 | 4 | 48% | 34% | 4 | 0 | 1 | 0 | single_population |
| field001 | **max 赤1000** | 104 | 68 | 18 | 17 | 284 | 10 | 33% | 34% | **0** | 1 | 1 | 0 | single_population |

細胞ごとの前後対応（緑のみ → max 赤1000、体積の重なりで対応付け）:
- field000: 緑のみの mCherry 17 個は 17 個とも対応が付き、Z 範囲・体積・Z 端の有無が**全部同じ**（Z 範囲の中央値 9 → 9 µm、2 µm 超の伸縮 0）。新しい細胞は 1 個（23 µm³ の小片）。field000 の赤の細胞は元から緑の漏れ込みで見えていたので、差が出ないのは想定どおり
- field001: 17 個とも対応が付き、Z 範囲の中央値 8 → 8 µm。**赤だけの明るい細胞（赤の中央値 972、524 µm³、Z 6 枚、Z 端に触れない）が新たに分割された**（緑のみでは輪郭なし。max p99 では別の細胞と混ざっていた）。3 個（元の Z 範囲 14 / 12 / 8 µm）が端の 1 面（芯の 15〜25% の明るさ）を得て Z 端に触れるようになった。1–99 のときの「全 11 面に伸びる」ではない
- 緑の合体は各視野 1 件で、どちらも小片（56 µm³・170 µm³）が隣の大きい緑細胞に吸収されたもの。2 つの本物の細胞の融合ではない。「消えた緑」は 0（max p99 では 5 / 4）
- max p99 が field001 で拾っていた暗い赤だけの細胞（赤の中央値 52〜61、緑 19〜156）は **max 赤1000 では拾わない**（0.05〜0.06 になる）。設計上の優先順位（Z 伸びの回避 > 暗い赤）
- mixing_index が field000 で 1.07 → 0.57 なのは、有効な異種接触は 2 で同じまま、赤どうしの接触が 1 つ増えて期待値（`expected_heterotypic_fraction` 0.117 → 0.208）が上がったため。計測側の式の話で分割の劣化ではない

### 目で確かめたこと（`review/*_compare.png` の outlines パネル）

- field001: 緑のみで輪郭の無かった真っ赤な細胞（大きな 1 個を含む）に max 赤1000 で輪郭が付いた。緑の細胞の輪郭は緑のみとほぼ同じ。max p99.9 で見えた「芯と縁」の二重輪郭は無い
- field000: 緑のみとほぼ同じ。暗い緑の細胞 1 個の中心に小さい別ラベルが乗った箇所が 1 つ増えた（緑のみでも別の細胞に同じ現象が 1 つある。膜局在の暗い細胞で Cellpose が内側を別物にする既知の癖で、今回の変更に固有ではない）
- field020（赤が最も疎な視野、`onefield020_abs1000/`）: 赤だけの細胞 4〜5 個に輪郭。mCherry 11 個、Z 範囲中央値 7 µm、Z 端 2/11、体積中央値 420 µm³。GFP 60 個、Z 端 18/60（他の視野と同程度）

### 合否

**合格**とした。根拠: 赤だけの明るい細胞が拾える（field001・field020）、赤の Z 伸びが無い（対応する mCherry 細胞の Z 範囲が緑のみと同じ）、緑が悪化しない（消えた緑 0、二重輪郭なし）、field000 の有効な異種接触 2 が保たれた。
達成していないこと: field001 の `single_population`（生の異種接触 1 つの緑側が Z 端に触れている。正しい分割でも 0 になる例）。暗い赤だけの細胞の検出。

### 時間

- max 赤1000 の 2 視野: 759 秒（pytest と一部同時）= 1 視野 約 6.3 分。field020 単独: 537 秒
- 全 25 視野: 13:47 JST 開始（結果は下に追記）
