# lambda/index.py
import json
import os
import boto3
import re
import urllib.request
import time
from botocore.exceptions import ClientError


# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
# MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
MODEL_ID = "https://70cf-34-87-47-66.ngrok-free.app"

# FastAPI呼び出し関数（タイムアウト設定付き）
def call_fastapi_with_timeout(url, payload, timeout=60):
    """FastAPIエンドポイントを呼び出す関数（タイムアウト設定あり）"""
    print(f"Calling FastAPI with timeout {timeout} seconds: {url}")
    
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    
    print("Sending request to FastAPI:", json.dumps(payload))
    response = urllib.request.urlopen(request, timeout=timeout)
    response_body = json.loads(response.read().decode('utf-8'))
    print("FastAPI response:", json.dumps(response_body, default=str))
    
    return response_body

def lambda_handler(event, context):
    # 残り実行時間を計算するための開始時間を記録
    start_time = time.time()
    
    try:
        # コンテキストから実行リージョンを取得し、クライアントを初期化
        global bedrock_client
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client('bedrock-runtime', region_name=region)
            print(f"Initialized Bedrock client in region: {region}")
        
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        print("Using model:", MODEL_ID)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # FastAPI呼び出しかBedrockのinvoke_modelどちらを使用するか判断
        if MODEL_ID.startswith("http"):
            # タイムアウト管理: Lambdaの残り実行時間を確認
            # Lambda関数のデフォルトタイムアウトは30秒
            # 既に5秒以上経過している場合、タイムアウトリスクを軽減
            elapsed_time = time.time() - start_time
            remaining_time = max(55 - elapsed_time, 5)  # 最低5秒は確保
            print(f"Elapsed time: {elapsed_time:.2f}s, Remaining time for API call: {remaining_time:.2f}s")
            
            # FastAPI呼び出し用のリクエストペイロード
            # 会話履歴から完全なプロンプトを作成
            full_prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    full_prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    full_prompt += f"Assistant: {msg['content']}\n"
            
            # 最後に応答を求めるプロンプトを追加
            full_prompt += "Assistant: "
            
            # FastAPI形式に合わせたリクエストペイロード
            # max_new_tensを小さくして高速化
            fastapi_payload = {
                "prompt": full_prompt,
                "max_new_tokens": 16,  # トークン数を減らして高速化
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # 履歴が長い場合は、プロンプトを短くする
            if len(messages) > 5:
                print("Long conversation history detected. Truncating prompt.")
                # 最新の2つのやり取りだけを保持
                recent_messages = messages[-5:]
                
                # 短縮版のプロンプトを作成
                short_prompt = ""
                for msg in recent_messages:
                    if msg["role"] == "user":
                        short_prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        short_prompt += f"Assistant: {msg['content']}\n"
                
                short_prompt += "Assistant: "
                fastapi_payload["prompt"] = short_prompt
            
            url = f"{MODEL_ID}/generate"
            
            # タイムアウト設定付きでAPIを呼び出し
            response_body = call_fastapi_with_timeout(
                url=url,
                payload=fastapi_payload,
                timeout=remaining_time  # 残り時間に基づいてタイムアウトを設定
            )
            
            # レスポンス形式に合わせて抽出
            generated_text = response_body.get('generated_text', "応答が見つかりません")
        else:
            # Bedrock用のリクエストペイロード
            bedrock_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    bedrock_messages.append({
                        "role": "user",
                        "content": [{"text": msg["content"]}]
                    })
                elif msg["role"] == "assistant":
                    bedrock_messages.append({
                        "role": "assistant", 
                        "content": [{"text": msg["content"]}]
                    })
            
            bedrock_payload = {
                "messages": bedrock_messages,
                "inferenceConfig": {
                    "maxTokens": 512,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            
            # Bedrock invoke_model APIを呼び出し
            print("Calling Bedrock invoke_model API with payload:", json.dumps(bedrock_payload))
            response = bedrock_client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(bedrock_payload),
                contentType="application/json"
            )
            
            # レスポンスを解析
            response_body = json.loads(response['body'].read())
            print("Bedrock response:", json.dumps(response_body, default=str))
            
            # 応答の検証
            if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
                raise Exception("No response content from the model")
            
            # アシスタントの応答を取得
            generated_text = response_body['output']['message']['content'][0]['text']
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": generated_text
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": generated_text,
                "conversationHistory": messages
            })
        }
        
    except Exception as error:
        print("Error:", str(error))
        
        # タイムアウトエラーかどうかを確認
        error_str = str(error)
        if "timed out" in error_str.lower():
            error_message = "APIリクエストがタイムアウトしました。処理に時間がかかりすぎています。"
            suggestion = "会話履歴を短くするか、より短い質問を試してください。"
        else:
            error_message = str(error)
            suggestion = "再試行するか、管理者にお問い合わせください。"
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": error_message,
                "suggestion": suggestion
            })
        }