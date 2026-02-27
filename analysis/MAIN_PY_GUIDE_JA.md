# analysis/main.py 詳細ドキュメント（日本語版）

本ドキュメントは、[main.py](main.py) の最新版実装について、
「何を計算しているか」「どの手法を使っているか」「何のために使うか」を、開発者・研究者向けに整理したものです。

---

## 1. このコードの目的

[main.py](main.py) は 3D 体操動作解析のメインエントリで、主に以下を行います。

1. 各人物の融合済み 3D キーポイント列を読み込む。
2. 3つの基本指標を計算する（ツイスト / 姿勢 / 脱力）。
3. 時系列の「不連続・不平滑」な点をロバストに除去する。
4. 派生指標を計算する（周期振幅、絶対逸脱、正面時手首位置）。
5. 個人ごとの JSON・可視化画像を保存する。
6. 全人物の比較 JSON と CSV を保存する。

---

## 2. 入出力と依存モジュール

### 2.1 依存モジュール

- [analysis/metrics.py](metrics.py)
  - `compute_twist`
  - `compute_trunk_tilt`
  - `compute_wrist_lead_angle`
  - `MHR70_INDEX`
- [analysis/visualize.py](visualize.py)
  - `plot_all_visualizations`
  - `plot_derived_metrics_summary`
- [fuse/load.py](../fuse/load.py)
  - `load_fused_sequence`

### 2.2 既定のパス

- 入力: `/workspace/code/logs/fuse/person_*`
- 出力: `/workspace/code/logs/analysis`

---

## 3. 計算内容（指標定義）

### 3.1 基本時系列指標

1. **ツイスト（Twist）**
   - 肩と股関節の回旋差。
   - `unwrap` 後に度数へ変換して利用。

2. **姿勢（Trunk Tilt）**
   - 体幹の鉛直軸に対する傾き角。
   - 度数へ変換して利用。

3. **脱力（Wrist Lead）**
   - 遅れ手の手首角度（体幹基準）。
   - `unwrap` 後に度数へ変換して利用。

### 3.2 派生指標（今回の主対象）

1. **ツイスト派生**
   - 旧定義: `局所max平均 - 局所min平均`
   - 新定義（主使用）: `隣接極値ペア振幅の平均`（`cycle_amp_mean_deg`）

2. **姿勢派生**
   - `|tilt|` の平均（鉛直軸からの絶対逸脱度平均）

3. **脱力派生**
   - 肩が正面を向いたフレームで、遅れ手手首位置 `XYZ` の平均
   - 比較では主に `Y` 成分を使用

---

## 4. 使用手法（アルゴリズム）

### 4.1 不平滑点の除去

関数: `_filter_unsmooth_signal(values, median_window=9, z_thresh=3.5, smooth_window=5)`

処理手順:

1. 中央値フィルタでベースライン生成（`_median_filter`）
2. 残差を計算
3. MAD ベースのロバスト z スコアで外れ値判定
4. 外れ値を NaN 化
5. 線形補間で補完（`_interpolate_nan`）
6. 移動平均で軽く平滑化（`_moving_average`）

効果:
- スパイクや急な飛びを抑制
- 主要トレンドを保持
- ピーク計算や統計の安定化

### 4.2 ツイストのピーク計算

関数: `_compute_twist_peak_gap_deg(twist_deg)`

同時に2種類の値を出力:

1. **legacy**
   - 局所極大・局所極小を抽出
   - `peak_gap_legacy_deg = mean(max) - mean(min)`

2. **cycle（主使用）**
   - 極値インデックスを時系列順に並べる
   - 種類が異なる隣接極値をペア化
   - 各ペアの振幅 `abs(v_{i+1} - v_i)` を平均

### 4.3 正面判定と手首位置

- `_compute_shoulder_frontal_mask`
  - 肩ラインの水平投影ベクトルと参照軸の角度で判定
  - `0` 付近と `π` 付近を正面とみなす
- `_compute_frontal_wrist_position`
  - 正面フレームで遅れ手手首 `XYZ` を平均化

---

## 5. 処理フロー

### 5.1 `analyze_sequence`（単人）

1. データ読み込み (`kpts_world`, `fps`)
2. Twist / Tilt / Lead を算出
3. 各時系列を不平滑除去
4. 派生指標算出（Twist, 姿勢, 脱力）
5. 統計表示と除去件数表示
6. 出力保存
   - `analysis_{id}.json`
   - `analysis_{id}_derived_values.json`
7. 可視化出力
   - 通常4図
   - `analysis_{id}_derived_metrics.png`（時系列 + box plot）

### 5.2 `main`（全体）

1. `person_*` を列挙
2. 各人物に対して `analyze_sequence` 実行
3. 全体出力
   - `comparison_summary.json`
   - `derived_metrics_all_persons.csv`

---

## 6. 出力ファイルの意味

### 6.1 個人フォルダ `person_{id}`

- `analysis_{id}.json`
  - フィルタ後時系列、統計、派生指標本体
- `analysis_{id}_derived_values.json`
  - 3派生指標の要約値（後処理向け）
- `analysis_{id}_derived_metrics.png`
  - 派生指標の可視化（下段に box plot）

### 6.2 全体フォルダ

- `comparison_summary.json`
  - 指標ごとの高群/低群比較（中央値分割）
- `derived_metrics_all_persons.csv`
  - 全人物の派生指標テーブル（Excel/pandas利用向け）

---

## 7. 各指標の実務的な意味

1. **Twist cycle amp mean**
   - 回旋振幅の大きさ・リズムの安定性を評価しやすい
2. **|Tilt| mean**
   - 鉛直からの逸脱の全体量を定量化
3. **Wrist mean at shoulder frontal**
   - 脱力タイミングを空間位置として比較可能
4. **Box plot**
   - 中央値・四分位・分散・外れ傾向を視覚的に確認可能

---

## 8. 主要パラメータと調整

既定値:
- `median_window=9`
- `z_thresh=3.5`
- `smooth_window=5`

調整の目安:
- 取りすぎる（平滑しすぎ）→ `z_thresh` を上げる、窓を小さく
- 取り足りない（ノイズ残る）→ `z_thresh` を下げる、窓を大きく

推奨プリセット:
- 保守: `z_thresh=4.5, median_window=7, smooth_window=3`
- 標準: `z_thresh=3.5, median_window=9, smooth_window=5`
- 強め: `z_thresh=2.8, median_window=11, smooth_window=7`

---

## 9. 注意点

1. 極値法はノイズ条件によって極値数が増減する。
2. `unwrap` は連続性向上に有効だが、角度レンジの見え方は変わる。
3. `frontal_mask`（lead算出側）と `shoulder_frontal_mask`（肩向き判定）は別概念。
4. `shoulder_frontal_frame_count` が少ない場合、手首平均値の信頼性は低下する。

---

## 10. 実行方法

```bash
python code/analysis/main.py
```

実行後チェック:
- 個人フォルダに `analysis_{id}_derived_values.json` があるか
- ルートに `comparison_summary.json` と `derived_metrics_all_persons.csv` があるか
- `analysis_{id}_derived_metrics.png` に box plot が描画されているか

---

## 11. 関数インデックス

- 平滑化・外れ値処理: `_filter_unsmooth_signal`
- Twist派生: `_compute_twist_peak_gap_deg`
- 正面判定: `_compute_shoulder_frontal_mask`
- 手首平均: `_compute_frontal_wrist_position`
- 単人派生保存: `_save_person_derived_values`
- 全体CSV保存: `_save_all_persons_derived_csv`
- 全体比較保存: `_save_comparison_summary`
- メイン処理: `analyze_sequence`, `main`
