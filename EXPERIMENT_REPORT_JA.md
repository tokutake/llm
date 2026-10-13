# 日本語ミニGPT実験レポート

実施日: 2026-10-13

## 目的

2019年のIntel iMacで、青空文庫の日本語小説を使った小型GPTを事前学習し、日本語らしい文章が生成されるか試した。

## データと出典

- 作品: 夏目漱石『吾輩は猫である』、青空文庫 図書カード [No.789](https://www.aozora.gr.jp/cards/000148/card789.html)
- 底本: 『夏目漱石全集1』ちくま文庫、筑摩書房、1987年
- 青空文庫ファイル: `789_ruby_5639.zip`（Shift-JIS、2018-02-05更新）
- 入力: 柴田卓治。校正: 田尻幹二、高橋真也、しず、瀬戸さえ子、おのしげひこ、渡部峰子。
- 前処理: 第1章から物語本文を取り出し、青空文庫のルビと編集注記を除去。旧仮名遣い等の本文表記は現代化していない。
- 前処理後: 321,139文字。コーパス本文はリポジトリに含めず、[`prepare_aozora.py`](prepare_aozora.py) で取得・生成する。

青空文庫の取り扱い規準では、著作権が切れた作品は有償・無償を問わず複製・再配布・翻案でき、出典情報を残すことが望まれている。[規準](https://www.aozora.gr.jp/guide/kijyunn.html)

## モデルと学習

- BPE語彙数: 4,096
- BPE化後のコーパス: 220,797 tokens
- GPT: 2層、埋め込み幅96、4 attention heads、context 96
- パラメータ数: 626,304
- batch size: 16
- optimizer: AdamW、learning rate 0.001
- training steps: 2,000
- device: CPU
- 学習時間: 134.9秒（約14.8 steps/秒）
- tokenizer学習時間: 約83.2秒
- loss: step 1で8.339、最後の100 step平均で3.993

## Kaggle GPU 実験

実施日: 2026-10-13。上記のローカル実験と同じ前処理済み本文、BPE tokenizer、モデル構成、seed、batch、context、学習率、2,000 stepを使い、Kaggle Notebookで実行した。Notebookは[T4 x2](https://www.kaggle.com/code/tokutake/notebook63ce7bb1f6)を選択したが、学習ログの `torch.cuda.get_device_name(0)` は `Tesla T4` と表示された。したがって、この測定はGPU 2枚で並列学習した結果ではなく、1枚のT4上での結果である。

- 実行環境: Kaggle、PyTorch 2.11.0+cu128、CUDA available: True、Accelerator設定: GPU T4 x2、実際に使ったdevice: cuda / Tesla T4
- 学習データ: 321,139文字、220,797 BPE tokens、語彙4,096
- モデル: 626,304 parameters
- 学習時間: スクリプト計測11.1秒。KaggleのNotebookセル表示は32.943秒（セル起動等を含む）
- loss: step 1で8.339、step 2,000で3.993
- 出力: `wagahai_torch.pt` を保存。生成もNotebookで実施

KaggleのNotebook出力には以下の生成が記録された。

```text
吾輩は猫である。名前はまだ無い。主人が書斎の中へ這入って来て何でも見給え。
　この主人の主人の頭と、ちょっと弁を振り出す。この通り台所へ乗って見ると、吾輩は、眼を突けておくふむして「それから、手はもう少し、ぐんすまかりますから、僕もや、どうするさ。もう少しやっとおしまい、僕の手で実験を買ってるのよ」と聞く。「そのそいつは少し気が大切り込めてしまった。僕
```

猫・主人・書斎など本文由来らしい語や語調は出るが、文意は破綻している。単一作品を小型モデルで2,000 step学習したデモであり、自然な文章生成を示すものではない。

## Kaggle T4 とローカルCPUの比較

ローカルCPUで同じ2,000-step学習にかかった134.9秒と、Kaggle T4の学習スクリプト内計測11.1秒を比較すると、今回の設定ではT4が約12.2倍高速だった（123.8秒短縮、約91.8%減）。Notebookセル全体の32.943秒を使うと約4.1倍。何を「時間」とするかで値が変わるため、純粋な学習ループ同士の目安は約12倍、Notebookのセル経過時間を含めた比較は約4倍と記す。別マシン・別フレームワークの測定であり、一般的なGPU対CPU比ではない。

| 実行場所 / device | 2,000 step | 最後の100 step平均loss | 備考 |
|---|---:|---:|---|
| 2019 iMac / Intel Core i5-8500 CPU | 134.9秒 | 3.993 | ローカル実測 |
| Kaggle / Tesla T4 1基 | 11.1秒 | 3.993 | 学習スクリプト計測 |
| Kaggle / Tesla T4 1基 | 32.943秒 | 3.993 | Notebookセルの実行時間表示 |

## TFLOPS の読み方

NVIDIAの公称仕様ではTesla T4 1基あたり FP32 は8.1 TFLOPS、Tensor Coreを使う混合精度FP16/FP32は最大65 TFLOPS。理論ピーク値で、今回の学習で実測したFLOPSではない。学習コードはFP32の通常のPyTorch演算で、混合精度やTensor Core向け最適化を明示的に使っていないため、65 TFLOPSを今回の実効性能として扱えない。[NVIDIA T4仕様](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)

仮に設定どおり2基を並べて理論値を足すならFP32 16.2 TFLOPS、混合精度最大130 TFLOPSだが、今回のログには1基のT4しか現れず、その合算値を実験結果にしてはいけない。また、i5-8500 CPUの理論TFLOPSや今回の実効FLOPSは測定していない。CPUとの比較は同一スクリプトの経過時間に基づく約12.2倍（学習ループ計測）/約4.1倍（Notebookセル時間）であり、TFLOPS比ではない。

TFLOPSは1秒あたりの浮動小数点演算を兆回単位で表した理論ピーク指標。1 TFLOPS = 10^12 FLOP/s。モデル規模やバッチが小さい場合は、メモリ転送、カーネル起動、データ準備などの影響でピーク性能に届かない。このため今回のような小さなモデルでは、仕様のTFLOPSから実時間を直接予測できない。

lossは学習中の値で、独立した評価用コーパスは使っていない。文章理解や未知文への汎化を示す数値ではない。

## 実行環境

- iMac19,1（2019）
- CPU: Intel Core i5-8500、6コア、3.0 GHz
- GPU: AMD Radeon Pro 570X、4 GB VRAM、Metal 2
- メモリ: 32 GB
- macOS 15.7.9、Python 3.11.15、PyTorch 2.2.2
- PyTorch MPS: 利用可能

GPUの理論ピークTFLOPSはこの実験では測定していない。理論ピーク値と、この小型モデルでの実効速度は別の指標として扱う。

## CPU / MPS 短時間比較

同じ日本語データ、モデル設定、seedで各100 stepを1回実行した。MPSの初回起動時間を含む。

| device | 100 step | 最後の100 step平均loss |
|---|---:|---:|
| CPU | 6.5秒 | 6.762 |
| MPS (Radeon Pro 570X) | 7.0秒 | 6.762 |

この設定ではCPUが約8%速かったが、1回・100 stepの短い測定なので小さな差として見る。2,000 stepの本学習はCPUで行った。

## 生成例

再現コマンド:

```sh
.venv311/bin/python generate_torch.py \
  --device cpu \
  --checkpoint wagahai_torch.pt \
  --tokenizer wagahai_tokenizer.json \
  --prompt "吾輩は猫である。名前はまだ無い。" \
  --tokens 120 --temperature 0.8 --top-k 20 --seed 20261013
```

**Prompt**

> 吾輩は猫である。名前はまだ無い。

**Output**

```text
吾輩は猫である。名前はまだ無い。元来今の所有名の人間がいかに普通の人を見做して人が分らぬ事は出来ない。人は、逆上家の逆上である。主人の不安に対の頭を挟むに対して、吾人の心を与える。彼の大に、これらこの家の主人の大に対して奇問を有している。
「君は全くないの名産でも知しなさらん、僕が僕も実はそう思いますので大丈夫と云うじゃないか」と主人は例のごとく例の鼻
```

単語や表記には原作らしさが見えるが、文法や意味のつながりは崩れている。これは約32万字の単一作品で学習した小型モデルのデモであり、実用的な日本語モデルではない。
