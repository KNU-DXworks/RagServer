from flask import Flask, request, jsonify
from google.cloud import bigquery
import json
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = Flask(__name__)
client = bigquery.Client()
dataset_id = "dxworks-rag-ai.inbody_vectors"
table_id = "inbody_vector"

#BigQuery 테이블 경로 지정
TABLE_ID = f"{dataset_id}.{table_id}"

@embedding.route("/api/gemini/embeddinginbody",methods = ['POST'])
def inbody_data():
    try:
        #POST 요청해서 JSON DATA 파싱.
        data = request.get_json()
        if not data:
            return jsonify({"error" : "No Json payload provided"}) ,400

        #임시 파일에 JSON 형식으로 저장.
        with tempfile.NamedTemporaryFile(mode="w+",delete=False,suffix=".json") as temp_file:
            for item in [data]: #여러개면 리스트
                json.dump(item,temp_file)
                temp_file.write('\n')
            temp_file.flush()

            #BigQuery에 업로드
            job_config = bigquery.LoadJobConfig(
                source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                write_disposition = "WRITE_APPEND",
                autodetect = True,
            )

            with open(temp_file.name,"rb") as source_file:
                load_job = client.load_table_from_file(source_file,TABLE_ID,job_config=job_config)

            load_job.result() #완료대기

            return jsonify({"status" : "success", "message" : "BigQuery에 데이터 삽입 성공."}), 200
        
    except Exception as e:
        return jsonify({"error" : str(e)}), 500
        

@embedding.route("/api/main/recommend", methods=["POST"])
def recommend_similar_inbodies():
    try:
        req_data = request.get_json()
    
        goal_vector = req_data.get("goal_vector")
        print(goal_vector)
        
        if not goal_vector:
            return jsonify({"error": "goal_vector가 비어있습니다."}), 400

        # BigQuery에서 전체 벡터 가져오기
        query = f"SELECT userId, vector FROM `{TABLE_ID}`"
        results = client.query(query).result()

        vectors = []
        user_ids = []

        for row in results:
            if row.vector:
                user_ids.append(row.userId)
                vectors.append(row.vector)

        # numpy로 변환
        all_vectors = np.array(vectors)
        goal_vector_np = np.array(goal_vector).reshape(1, -1)

        # 코사인 유사도 계산
        similarities = cosine_similarity(goal_vector_np, all_vectors)[0]

        # 유사도 top3
        top_indices = np.argsort(similarities)[-10:][::-1]
        top_users = [
            {"userId": int(user_ids[i]), "similarity": float(similarities[i])}
            for i in top_indices
        ]

        return jsonify({"top10": top_users}),200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#더미 인바디 데이터 넣기.
@embedding.route("/api/main/put/inbodydata", methods=["POST"])
def insert_inbody_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        user_id = data.get("userId")
        vector = data.get("vector")

        if user_id is None or vector is None:
            return jsonify({"error": "userId와 vector는 필수입니다."}), 400

        # BigQuery에 삽입할 행 구성
        rows_to_insert = [{
            "userId": user_id,
            "vector": vector
        }]

        errors = client.insert_rows_json(TABLE_ID, rows_to_insert)

        if errors == []:
            return jsonify({"status": "success", "message": "BigQuery에 새로운 데이터 삽입 성공"}), 200
        else:
            return jsonify({"status": "error", "errors": errors}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    embedding.run(host = "127.0.0.1", port=5000)
    