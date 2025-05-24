from flask import Flask, request, jsonify,send_file,after_this_request
from google.cloud import bigquery
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

visualize = Flask(__name__)
client = bigquery.Client()
dataset_id = "dxworks-rag-ai.inbody_vectors"
table_id = "inbody_vector"

#BigQuery 테이블 경로 지정
TABLE_ID = f"{dataset_id}.{table_id}"

def fetch_vectors():
    query = f"SELECT userId, vector FROM `{TABLE_ID}`"
    results = client.query(query).result()
    user_ids = []
    vectors = []
    for row in results:
        if row.vector:
            user_ids.append(row.userId)
            vectors.append(row.vector)
    return user_ids, np.array(vectors)



def generate_3d_plot(vectors,goal_vector=None):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(reduced[:,0],reduced[:,1],reduced[:,2],label = 'Users', c = 'blue')

    if goal_vector is not None:
        goal_reduced = pca.transform([goal_vector])
        ax.scatter(goal_reduced[:,0],goal_reduced[:,1],goal_reduced[:,2],label = 'Goal',c='red',marker = 'x',s=100)

    ax.set_title("3D Visualization of inbody vectors")
    ax.legend()

    #임시 이미지 저장.
    temp_file = tempfile.NamedTemporaryFile(delete=False,suffix=".png")
    plt.savefig(temp_file.name)
    plt.close(fig)
    return temp_file.name

@visualize.route("/api/main/visualize",methods=["POST"])
def visualize_vectors():
    try:
        data = request.get_json()
        goal_vector = data.get("goal_vector") if data else None

        _,vectors = fetch_vectors()
        image_path = generate_3d_plot(vectors,goal_vector)
        
        @after_this_request
        def remove_file(response):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"파일 삭제 실패 : {e}")
            return response

        return send_file(image_path,mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
   
if __name__ == "__main__":
    visualize.run(host='127.0.0.1',port=5001,debug=True)
    