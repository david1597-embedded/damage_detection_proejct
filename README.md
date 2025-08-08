# 🚗 자동차 외관 손상 탐지 시스템 개발 프로젝트
 
![rect](https://capsule-render.vercel.app/api?type=rect&color=0:6a11cb,100:2575fc&text=Vehicle%20Damage%20Detection&fontAlign=50&fontSize=40&fontColor=ffffff&textBg=true&desc=EfficientDet%20D0%20%2B%20SSD%20MobileNet%20V2&descAlign=50&descAlignY=70&descSize=20&descColor=eeeeee)


![Detection Model](https://img.shields.io/badge/Classification-SSD%20MobileNet%20V2-green)
![Language](https://img.shields.io/badge/Language-Python-yellow)
![Platform](https://img.shields.io/badge/Platform-TensorFlow%202.x-orange)

---

# 🔧프로젝트 개요
- 객체 탐지 모델을 활용한 자동차 외관 손상 탐지 시스템
- EfficientDet D0와 SSD MobileNet V2를 사용해 두 모델의 정확도를 비교
- PyQt5를 활용해 이미지 전처리와 자동모드를 통한 여러 이미지들을 자동으로 분석해주는 기능 제공

---

# 💡 UseCase

---

# 📁 Work Folder

├── 📁 images                 # 예측을 진행할 자동차 외관 이미지\
│   ├── images\
│   └── json\
│\
├── 📁 models                 # 사전 학습된 모델 저장 디렉토리\
│   ├── efficientdet\
│   │   └── saved_model\
│   │       ├── assets\
│   │       ├── variables\
│   │       │   ├── variables.data-00000-of-00001\
│   │       │   └── variables.index\
│   │       └── saved_model.pb\
│   │\
│   ├── ssdmobilenet\
│       └── saved_model\
│           ├── assets\
│           ├── variables\
│           │   ├── variables.data-00000-of-00001\
│           │   └── variables.index\
│           └── saved_model.pb\
│\
├── 📁 results\
│   ├── model_result\
│   │   └── 모델 별 결과와, 전체결과에 대한 분석이 csv 파일로 저장됨\
├── 🐍 main.py               # 실행 파일 (예측 실행용)\





[📥 모델 다운로드 링크](https://drive.google.com/file/d/1RqXMzyd_pvzwulrBYhJtmWC1bSAP_3Of/view?usp=drive_link)  
> 현재 작업 디렉토리에 그대로 추가

---

## 🖼️ ScreenShot

### 🖱️ GUI (Manual Mode)
<img width="1394" height="737" alt="image" src="https://github.com/user-attachments/assets/030dc1f8-bddf-4dd8-b063-c739768e84db" />

---

### ⚡ GUI (Automatic Mode)
<img width="1427" height="539" alt="image" src="https://github.com/user-attachments/assets/385c795d-a508-4681-96dc-ed9bd9cd9dba" />

---

### 📂 Select work folder
<img width="1437" height="789" alt="image" src="https://github.com/user-attachments/assets/6aca6a83-683d-4d36-a6c6-91ecf5643f02" />

---

### 🤖 Automatic prediction
<img width="503" height="308" alt="image" src="https://github.com/user-attachments/assets/9db53119-9836-4a87-be75-5b9b3e56985b" />

---

## 📊 Result

### 📌 모델 별 예측 결과 저장
<p align="center">
  <img src="https://github.com/user-attachments/assets/67839aa4-9104-4a66-b336-5223789f2d0d" width="45%" />
  <img src="https://github.com/user-attachments/assets/59df7ddd-e2bd-4e61-91d3-edd5ef6aaeb5" width="45%" />
</p>

- 모델별 예측 결과에 대해서 저장. 작업 이미지 경로와 불량 판단 여부 저장.
- 불량 개소가 1개라도 탐지된다면 불량품임(모델의 정확도가 낮아 0.4를 기준으로 함)

---

### 📈 전체 결과
<img width="1595" height="278" alt="image" src="https://github.com/user-attachments/assets/1bb13863-fc76-4e57-a57d-5bfbc3829c60" />

- 모델오차와 정답률을 저장해 두 모델간의 탐지 결과를 비교함

---

## ⚙️ Installation


```
git clone [https://github.com/david1597-embedded/game_recommendation.git](https://github.com/david1597-embedded/damage_detection_proejct.git)
cd damage_detection_project
```

### 💻Windows

```
python -m venv venv
venv\Sciprts\activate
pip install -r requirements.py
python main.py
```
### 🐧Ubuntu
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.py
python main.py
```



