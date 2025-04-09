
"""
Generate training data for fine-tuning a Japanese LLM for test case generation.
This script creates a JSONL file with Japanese user stories and corresponding test cases.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "training_data.jsonl"

USER_STORY_TEMPLATES = [
    "ユーザーとして、{action}ができるようにしたい。それによって{benefit}。",
    "{role}として、{action}する機能が欲しい。なぜなら{benefit}から。",
    "{role}は{action}ができる必要がある。これにより{benefit}。",
    "{action}機能を実装して欲しい。{role}が{benefit}ために必要。",
    "私は{role}として、{action}したい。それは{benefit}ためだ。"
]

ROLES = [
    "管理者", "一般ユーザー", "ゲスト", "顧客", "営業担当者", "マネージャー", 
    "学生", "教師", "患者", "医師", "開発者", "デザイナー", "テスター",
    "会計士", "マーケティング担当者", "サポートスタッフ", "システム管理者"
]

ACTIONS = [
    "ログイン", "ログアウト", "アカウント登録", "パスワードリセット", 
    "プロフィール編集", "データ検索", "レポート作成", "ファイルアップロード", 
    "コメント投稿", "通知設定", "予約", "支払い", "商品追加", "商品削除",
    "ユーザー招待", "権限設定", "メッセージ送信", "カレンダー表示", 
    "タスク割り当て", "進捗確認", "承認", "拒否", "フィルタリング",
    "ソート", "エクスポート", "インポート", "バックアップ", "復元",
    "グラフ表示", "分析", "集計", "印刷", "共有", "ダウンロード"
]

ACTION_DETAILS = [
    "簡単に", "素早く", "安全に", "効率的に", "自動的に", "一括で", 
    "定期的に", "リアルタイムで", "オフラインでも", "モバイルから",
    "複数の条件で", "カスタマイズして", "詳細に", "視覚的に"
]

BENEFITS = [
    "時間を節約できる", "効率が上がる", "ミスを減らせる", "コストを削減できる",
    "顧客満足度が向上する", "売上が増加する", "業務プロセスが改善される",
    "意思決定が迅速になる", "コミュニケーションが向上する", "透明性が高まる",
    "セキュリティが強化される", "コンプライアンスを確保できる", "品質が向上する",
    "生産性が上がる", "ユーザー体験が向上する", "競争力が高まる",
    "情報共有が容易になる", "トラブルを早期に発見できる", "分析が容易になる"
]

TEST_CASE_TEMPLATES = [
    "• {condition}場合、{expected_result}こと。",
    "• {action_detail}時に、{expected_result}ことを確認する。",
    "• {condition}状態で{action_detail}と、{expected_result}こと。",
    "• {negative_condition}場合、{negative_result}こと。",
    "• {edge_case}の場合、システムは{edge_case_result}こと。"
]

CONDITIONS = [
    "有効な認証情報で", "無効な認証情報で", "必須フィールドがすべて入力された",
    "必須フィールドが未入力の", "権限を持つユーザーが", "権限のないユーザーが",
    "ネットワーク接続がある", "ネットワーク接続がない", "データが存在する",
    "データが存在しない", "入力値が最大長の", "入力値が最小長の",
    "特殊文字を含む", "日本語を含む", "英数字のみの", "数字のみの",
    "大量のデータがある", "システムに負荷がかかっている"
]

EXPECTED_RESULTS = [
    "正常に処理が完了する", "エラーメッセージが表示される", "確認メッセージが表示される",
    "データが保存される", "データが更新される", "データが削除される",
    "画面が遷移する", "変更が反映される", "ログが記録される",
    "通知が送信される", "処理が中断される", "処理が再開される",
    "ユーザーにフィードバックが提供される", "システムが適切に応答する",
    "セッションが維持される", "セッションが終了する"
]

NEGATIVE_CONDITIONS = [
    "無効なデータで", "サーバーがダウンしている", "タイムアウトが発生した",
    "同時に複数のリクエストが発生した", "不正なアクセスを試みた",
    "サポートされていない形式のファイルを使用した", "容量制限を超えた",
    "互換性のないバージョンを使用した", "破損したデータを使用した"
]

NEGATIVE_RESULTS = [
    "適切なエラーメッセージが表示される", "データの整合性が保たれる",
    "セキュリティ違反が検出される", "システムが安全に失敗する",
    "ユーザーに適切なガイダンスが提供される", "回復手順が実行される",
    "管理者に通知が送信される", "エラーがログに記録される"
]

EDGE_CASES = [
    "極端に大きな値", "極端に小さな値", "ゼロ値", "負の値",
    "境界値", "空の文字列", "非常に長い文字列", "国際文字",
    "重複するデータ", "非常に古いデータ", "将来の日付"
]

EDGE_CASE_RESULTS = [
    "適切に処理する", "明確なエラーメッセージを表示する",
    "デフォルト値を使用する", "値を切り捨てる", "値を丸める",
    "処理をスキップする", "代替処理を実行する"
]

def generate_user_story():
    """Generate a random Japanese user story."""
    template = random.choice(USER_STORY_TEMPLATES)
    role = random.choice(ROLES)
    action = f"{random.choice(ACTIONS)}を{random.choice(ACTION_DETAILS)}"
    benefit = random.choice(BENEFITS)
    
    return template.format(role=role, action=action, benefit=benefit)

def generate_test_cases():
    """Generate 2-3 test cases for a user story."""
    num_test_cases = random.randint(2, 3)
    test_cases = []
    
    templates = random.sample(TEST_CASE_TEMPLATES, num_test_cases)
    
    for template in templates:
        if "{condition}" in template and "{expected_result}" in template:
            test_case = template.format(
                condition=random.choice(CONDITIONS),
                expected_result=random.choice(EXPECTED_RESULTS)
            )
        elif "{action_detail}" in template and "{expected_result}" in template:
            test_case = template.format(
                action_detail=random.choice(ACTION_DETAILS),
                expected_result=random.choice(EXPECTED_RESULTS)
            )
        elif "{negative_condition}" in template and "{negative_result}" in template:
            test_case = template.format(
                negative_condition=random.choice(NEGATIVE_CONDITIONS),
                negative_result=random.choice(NEGATIVE_RESULTS)
            )
        elif "{edge_case}" in template and "{edge_case_result}" in template:
            test_case = template.format(
                edge_case=random.choice(EDGE_CASES),
                edge_case_result=random.choice(EDGE_CASE_RESULTS)
            )
        else:
            test_case = template.format(
                condition=random.choice(CONDITIONS),
                expected_result=random.choice(EXPECTED_RESULTS),
                action_detail=random.choice(ACTION_DETAILS),
                negative_condition=random.choice(NEGATIVE_CONDITIONS),
                negative_result=random.choice(NEGATIVE_RESULTS),
                edge_case=random.choice(EDGE_CASES),
                edge_case_result=random.choice(EDGE_CASE_RESULTS)
            )
        
        test_cases.append(test_case)
    
    return "\n".join(test_cases)

def generate_training_data(num_samples=100):
    """Generate training data and save to JSONL file."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            user_story = generate_user_story()
            test_cases = generate_test_cases()
            
            sample = {
                "input": user_story,
                "output": test_cases
            }
            
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Generated {num_samples} training samples and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_training_data()
