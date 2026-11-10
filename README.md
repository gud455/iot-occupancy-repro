# 办公室占用检测复现

基于 UCI Occupancy Detection 的原始训练与测试划分，复现四种方法的分类结果和无人确认规则的时间回放。

使用 Python 3.12，在仓库根目录运行以下命令。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl --fail --location --output data/occupancy-detection.zip \
  'https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip'
unzip data/occupancy-detection.zip -d data
python reproduce.py
```

解压后，`data/` 中应有 `datatraining.txt`、`datatest.txt` 和 `datatest2.txt`。`provenance.json` 记录官方来源与 SHA-256 校验值。

运行后查看 `results.json`。例如，测试二的 `LDA-5` 准确率为 **98.76%**。脚本也会生成分类结果与时间回放的表格。

数据由 Luis Candanedo 发布，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。使用数据时请引用 [UCI Occupancy Detection（DOI 10.24432/C5X01N）](https://doi.org/10.24432/C5X01N)。
