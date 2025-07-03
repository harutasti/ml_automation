# Kaggle Automation Agent 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kaggleコンペティションを完全自動化するMLパイプラインツール。データのダウンロードから提出まで、すべてを自動化します。

## 📋 目次

- [特徴](#-特徴)
- [動作環境](#-動作環境)
- [インストール](#-インストール)
- [初期設定](#-初期設定)
- [使い方](#-使い方)
- [詳細な使用例](#-詳細な使用例)
- [カスタマイズ](#-カスタマイズ)
- [トラブルシューティング](#-トラブルシューティング)
- [開発者向け情報](#-開発者向け情報)

## ✨ 特徴

- 🚀 **完全自動化**: データダウンロードから提出まで全自動
- 🎯 **スマート介入**: 各ステージでの人間による介入が可能
- 📊 **自動EDA**: 探索的データ分析と可視化を自動実行
- 🔧 **高度な特徴量エンジニアリング**: 
  - 自動的な特徴量生成と選択
  - ターゲットエンコーディング（K-fold）
  - 頻度エンコーディング
  - クラスタリング特徴量
  - PCA/多項式特徴量
- 🤖 **複数アルゴリズム**: LightGBM、XGBoost、CatBoost等に対応
- 🔬 **Optunaによる最適化**: 高度なハイパーパラメータ探索
- 🎭 **高度なアンサンブル**: 
  - Voting、Stacking、Blending
  - マルチレベルスタッキング
  - ベイズモデル平均
- 🏷️ **自動タスク検出**: Kaggle APIメタデータから分類/回帰を自動判定
- 💾 **状態管理**: パイプラインの中断・再開が可能
- 🔌 **カスタムコード**: 任意のステージでコードを注入可能
- 🛡️ **セキュリティ機能**: コード注入時の安全性検証
- 📊 **実験追跡**: 全実験の詳細なログと比較機能

## 🖥️ 動作環境

- Python 3.8以上
- Linux/MacOS/Windows (WSL推奨)
- メモリ: 8GB以上推奨
- ディスク: 10GB以上の空き容量

## 📦 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/kaggle_automation.git
cd kaggle_automation
```

### 2. 環境構築（推奨: Conda）

#### 🐍 **方法A: Conda環境（推奨）**

```bash
# 環境ファイルから一括作成
conda env create -f environment.yml

# 環境を有効化
conda activate kaggle-agent

# 開発版としてインストール
pip install -e .
```

#### 🎯 **方法B: 手動でConda環境作成**

```bash
# 環境作成
conda create -n kaggle-agent python=3.9
conda activate kaggle-agent

# conda-forgeチャンネルを追加
conda config --add channels conda-forge

# 主要パッケージをcondaでインストール
conda install pandas numpy scikit-learn matplotlib seaborn
conda install lightgbm xgboost optuna tqdm pyyaml joblib click

# pipでしか入らないパッケージ
pip install kaggle catboost

# 開発版としてインストール
pip install -e .
```

#### 📦 **方法C: venv（従来方式）**

```bash
# venv環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -e .
```

### 3. インストール確認

```bash
# コマンドが使えることを確認
kaggle-agent --help

# Pythonから確認
python -c "import kaggle_agent; print('✓ Installation successful!')"
```

### 4. 開発者向け環境

```bash
# 開発用の充実した環境
conda env create -f environment-dev.yml
conda activate kaggle-agent-dev
pip install -e .
```

## 🔧 初期設定

### 1. Kaggle APIの設定

Kaggle APIトークンを取得して設定します：

1. [Kaggle](https://www.kaggle.com)にログイン
2. プロフィール → Account → Create New API Token
3. ダウンロードした`kaggle.json`を以下の場所に配置：

```bash
# Linux/MacOS
~/.kaggle/kaggle.json

# Windows
C:\Users\YourName\.kaggle\kaggle.json
```

```bash
# 権限設定（Linux/MacOS）
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 依存関係の確認

```bash
# すべての依存関係が正しくインストールされているか確認
python -c "import kaggle_agent; print('✓ Installation successful!')"
```

## 🚀 使い方

### 基本的な使用手順

#### 1. プロジェクトの初期化

```bash
# 新しいプロジェクトを作成
kaggle-agent init my-titanic-project --competition titanic

# 作成されたプロジェクトに移動
cd my-titanic-project
```

#### 2. 完全自動実行

```bash
# すべてのステージを自動実行
kaggle-agent run --full-auto
```

これにより以下が自動実行されます：
- データのダウンロードと読み込み
- 探索的データ分析（EDA）とレポート生成
- 特徴量エンジニアリング
- 複数アルゴリズムでのモデル訓練
- アンサンブルモデルの作成
- 予測ファイルの生成とKaggleへの提出

#### 3. 段階的実行

```bash
# 特定のステップのみ実行
kaggle-agent run --steps eda
kaggle-agent run --steps eda,features
kaggle-agent run --steps modeling,ensemble,submission
```

### コマンド一覧

```bash
# プロジェクト初期化
kaggle-agent init <project-name> --competition <competition-name>

# パイプライン実行
kaggle-agent run [OPTIONS]
  --full-auto        # 完全自動実行
  --steps <steps>    # 特定ステップのみ実行 (例: eda,features)
  --resume          # 前回の続きから実行

# 状態管理
kaggle-agent status              # 現在の状態を表示

# カスタムコード追加
kaggle-agent add-hook <stage> <file>  # カスタムコードの追加

# 結果確認
kaggle-agent results --summary   # 実行結果のサマリー表示
```

## 📚 詳細な使用例

### 例1: Titanicコンペティション（初心者向け）

```bash
# 1. プロジェクト作成
kaggle-agent init titanic-auto --competition titanic
cd titanic-auto

# 2. 設定を確認（必要に応じて編集）
cat config.yaml

# 3. 完全自動で実行
kaggle-agent run --full-auto

# 4. 結果を確認
ls -la data/submissions/
cat eda_output/eda_summary.md
```

### 例2: カスタム特徴量を追加

```bash
# 1. カスタム特徴量モジュールを作成
mkdir -p custom_modules
cat > custom_modules/my_features.py << 'EOF'
import pandas as pd

def create_family_features(df):
    """家族に関する特徴量を作成"""
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df
EOF

# 2. モジュールを登録
kaggle-agent inject my_features

# 3. 実行
kaggle-agent run --full-auto
```

### 例3: 特定ステージのみ実行

```bash
# EDAのみ実行
kaggle-agent run --stage eda

# モデリング以降を実行
kaggle-agent run --stage modeling --resume
```

### 例4: ハイパーパラメータのカスタマイズ

```yaml
# config.yaml を編集
optimization:
  hyperparameter_tuning:
    enabled: true
    method: optuna
    n_trials: 200  # 試行回数を増やす
    timeout: 7200  # 2時間に延長
```

## 🎨 カスタマイズ

### フックシステム

各ステージの前後でカスタムコードを実行できます：

```python
# hooks/after_eda.py
def after_eda_hook(pipeline, context):
    """EDA完了後に実行されるフック"""
    print("EDA完了！追加の分析を実行...")
    
    # カスタム分析を追加
    data = pipeline.data['train']
    custom_analysis = data.describe()
    
    # 結果を保存
    custom_analysis.to_csv('custom_analysis.csv')
    
    return {'additional_features': ['Feature1', 'Feature2']}
```

### カスタムモデルの追加

```python
# custom_modules/my_model.py
from sklearn.base import BaseEstimator, ClassifierMixin

class MyCustomModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # カスタムモデルの実装
        pass
    
    def predict(self, X):
        # 予測の実装
        pass
```

## 🔧 設定ファイル（config.yaml）

```yaml
project:
  name: my-project
  competition: titanic
  version: 0.1.0

pipeline:
  auto_mode: true
  stages:
    data_download:
      enabled: true
    eda:
      enabled: true
      generate_report: true
      visualizations:
        - correlation_heatmap
        - missing_values
        - target_distribution
    feature_engineering:
      enabled: true
      auto_generate: true
      methods:
        - numeric_transforms
        - categorical_encoding
        - interaction_features
    modeling:
      enabled: true
      algorithms:
        - lgbm
        - xgboost
        - catboost
        - random_forest
      cv_folds: 5
      validation_strategy: stratified
    ensemble:
      enabled: true
      methods:
        - voting
        - stacking
        - blending
    submission:
      enabled: true
      auto_submit: false  # 自動提出は無効

optimization:
  hyperparameter_tuning:
    enabled: true
    method: optuna
    n_trials: 100
    timeout: 3600

tracking:
  experiments: true
  save_all_models: true
```

## ❗ トラブルシューティング

### よくある問題と解決方法

#### 1. Kaggle認証エラー
```bash
# エラー: Kaggle API credentials not found
# 解決方法:
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. 認証が必要なコンペティション
```bash
# エラー: Competition requires acceptance of rules
# 解決方法:
# 1. Kaggleウェブサイトでコンペページにアクセス
# 2. "Join Competition"をクリックして規約に同意
# 3. その後、通常通りコマンドを実行
kaggle-agent init project-name --competition competition-name
```

#### 3. メモリ不足エラー
```yaml
# config.yaml で以下を設定
pipeline:
  data_processing:
    chunk_size: 10000  # データを分割処理
    reduce_memory: true  # メモリ最適化を有効化
```

#### 4. Conda環境でkaggle-agentコマンドが見つからない
```bash
# 解決方法1: 環境を再アクティベート
conda deactivate
conda activate kaggle-agent

# 解決方法2: フルパスで実行
~/yes/envs/kaggle-agent/bin/kaggle-agent run --full-auto

# 解決方法3: パッケージの再インストール
pip install -e . --force-reinstall
```

#### 5. 中断されたパイプラインの再開
```bash
# 最新のチェックポイントから再開
kaggle-agent run --resume
```

### デバッグモード

```bash
# 詳細なログを表示
export KAGGLE_AGENT_DEBUG=1
kaggle-agent run --full-auto

# または
kaggle-agent run --full-auto --verbose
```

## 👥 開発者向け情報

### プロジェクト構造

```
kaggle_agent/
├── cli.py                      # CLIエントリーポイント
├── pipeline.py                 # メインパイプライン
├── core/                       # コア機能
│   ├── auth.py                # 認証管理
│   ├── project.py             # プロジェクト管理
│   ├── competition.py         # コンペティション管理
│   ├── experiment_tracker.py  # 実験追跡
│   ├── state_manager.py       # 状態管理
│   └── code_injector.py       # コード注入
├── modules/                    # パイプラインモジュール
│   ├── eda.py                 # 探索的データ分析
│   ├── feature_engineering.py # 特徴量エンジニアリング
│   ├── modeling.py            # モデリング
│   ├── ensemble.py            # アンサンブル
│   └── submission.py          # 提出ファイル生成
└── hooks/                      # フックシステム
    └── hook_manager.py        # フック管理
```

### 開発環境のセットアップ

#### Conda環境（推奨）

```bash
# 開発用環境を作成
conda env create -f environment-dev.yml
conda activate kaggle-agent-dev

# 開発版としてインストール
pip install -e .

# pre-commitフックの設定
pre-commit install
```

#### 環境管理コマンド

```bash
# 環境のエクスポート
conda env export > environment-freeze.yml

# 環境の更新
conda env update -f environment.yml

# 環境の削除と再作成
conda deactivate
conda env remove -n kaggle-agent
conda env create -f environment.yml
```

#### テストとリント

```bash
# テストの実行
pytest tests/ -v --cov=kaggle_agent

# リンターの実行
black kaggle_agent/  # コードフォーマット
isort kaggle_agent/  # インポート整理
flake8 kaggle_agent/ # スタイルチェック
mypy kaggle_agent/   # 型チェック

# セキュリティチェック
bandit -r kaggle_agent/
safety check

# すべてのチェックを一括実行
pre-commit run --all-files
```

### 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- Kaggleコミュニティ
- 使用しているオープンソースライブラリの開発者の皆様

## 📧 お問い合わせ

- Issues: [GitHub Issues](https://github.com/yourusername/kaggle_automation/issues)
- Email: your-email@example.com

---

**Happy Kaggling! 🎉**