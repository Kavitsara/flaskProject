from flask import Flask, request, jsonify
import joblib  # ใช้สำหรับโหลดโมเดล
import numpy as np

app = Flask(__name__)

# โหลดโมเดลจากไฟล์ .joblib
model = joblib.load('/mnt/data/random_forest_model.joblib')


@app.route('/')
def hello_world():
    return 'Hello World!'


# สร้าง route สำหรับการคาดการณ์
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลที่ส่งมาเป็น JSON
        data = request.json
        # แปลงข้อมูลอินพุตเป็นรูปแบบ numpy array ที่โมเดลสามารถประมวลผลได้
        input_features = np.array(data['features']).reshape(1, -1)

        # ใช้โมเดลในการทำการคาดการณ์
        prediction = model.predict(input_features)

        # ส่งผลลัพธ์การคาดการณ์กลับไปในรูปแบบ JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

