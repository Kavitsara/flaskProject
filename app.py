from flask import Flask, request, jsonify
import joblib  # ใช้สำหรับโหลดโมเดล
import numpy as np

app = Flask(__name__)

# โหลดโมเดลจากไฟล์ .joblib
model = joblib.load('/mnt/data/random_forest_model.joblib')


# สร้าง route สำหรับการคาดการณ์ Weather Type
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลที่ส่งมาเป็น JSON
        data = request.json

        # ดึงข้อมูลจาก JSON และสร้าง input array
        temperature = data.get('Temperature')
        humidity = data.get('Humidity')
        wind_speed = data.get('Wind Speed')
        precipitation = data.get('Precipitation (%)')
        cloud_cover = data.get('Cloud Cover')
        uv_index = data.get('UV Index')
        season = data.get('Season')
        visibility = data.get('Visibility (km)')
        location = data.get('Location')

        # แปลงข้อมูลอินพุตเป็น numpy array เพื่อใช้ในการคาดการณ์
        input_features = np.array([[temperature, humidity, wind_speed, precipitation, cloud_cover,
                                    uv_index, season, visibility, location]])

        # ใช้โมเดลในการทำการคาดการณ์
        prediction = model.predict(input_features)

        # ส่งผลลัพธ์การคาดการณ์กลับไปในรูปแบบ JSON
        return jsonify({'Weather Type': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)