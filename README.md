# Spatial-Detection-
# 🛰️ ScanFlow - AI-powered Spatial Change Detection App

> 📱 Real-time room scanning & object change detection using YOLOv8 — optimized for mobile and edge devices.



https://github.com/user-attachments/assets/4d0c6370-ed5f-45bc-81fb-007036d250bc



![WhatsApp Image 2025-04-06 at 09 23 28_3452805e](https://github.com/user-attachments/assets/1ebdc191-7a61-4469-b344-40299ad6e21c)

---

## 🔍 What is ScanFlow?

**ScanFlow** is an AI-powered mobile application that helps track and detect spatial changes in dynamic environments such as **hotel rooms, warehouses, retail shelves, and office spaces**. It scans a room, builds a baseline map, and detects changes like **missing, moved, or newly added objects** in subsequent scans.

> 🎯 Designed for mobile, it empowers users with efficient, lightweight, and actionable spatial insights for **inventory, security, and space optimization**.

---

## ✨ Key Features

### ✅ Core Features
- **Initial Room Mapping** – Create a baseline by scanning the room and storing object layout.
- **Change Detection** – Detect & categorize new, removed, or repositioned items using YOLOv8.
- **Optimized Edge Processing** – Processes only changed areas to save resources.
- **Mobile/Edge Deployment** – Runs efficiently on low-power edge devices or phones.
- **User-Friendly Visualization** – Heatmaps, highlights & difference view overlays.

### 🧠 Bonus Features
- **Object Categorization & Alerts** – Triggers alerts when specific objects go missing or move.
- **Multi-Scan Analysis** – Tracks long-term changes and trends across multiple visits.
- **External System Integration** – Connects with inventory/security/facility software.
- **🗣️ Calling Agent (Smart Dirt & Damage Advisor)**  
  An AI-powered assistant that analyzes cleanliness/damage and **calls the manager** (voice/text) to **report critical issues**.  
  *Example alerts:*  
  > *"Room 104 has visible carpet stains and clutter. Immediate cleaning required."*  
  > *"Room 212 is clean. ✅ No issues detected."*

---

## 📲 UI/UX Flow

### 🛎️ Staff Login
- Streamlined check-in/checkout with photo comparison.
- AI detection for missing/damaged/moved items.
- Auto-generated issue reports.
- View room history & logs.

### 👨‍💼 Manager Login
- Review and verify AI alerts.
- Dashboard for trends and analytics.
- Confirm or override AI insights with context.
- Strategic decision-making support.
- Receives **automated calls/texts** from the **Calling Agent** for high-priority rooms.

---

## 🚀 Unique Selling Propositions (USPs)

- ⚡ **Real-Time Inventory Management**  
  Instant tracking of inventory movement and updates.
- 🎯 **Advanced Object Detection**  
  Fine-tuned YOLOv8n model for indoor/hotel-specific objects.
- 🔐 **Person Identification (Optional)**  
  Presence detection for security monitoring.
- 🧩 **Customizable Architecture**  
  Extendable to any space – from rooms to entire buildings.
- 🗣️ **Calling Agent for Actionable Alerts**  
  Voice/text AI assistant for instant room prioritization.

---

## ⚙️ Tech Stack

| Component            | Technology Used             |
|----------------------|-----------------------------|
| Object Detection     | YOLOv8n (fine-tuned)        |
| Backend              | Python, Flask/FastAPI       |
| Model Training       | Ultralytics, Roboflow       |
| Frontend/App         | React Native / Flutter      |
| Deployment           | Android, Raspberry Pi       |
| Visualization        | OpenCV, Matplotlib, Gradio  |
| Audio Alerts         | gTTS, pyttsx3, ElevenLabs   |
| Dataset              | Custom Hotel/Indoor Dataset |

---

## 📊 Performance Metrics

- **Model:** YOLOv8n (nano variant)
- **mAP@0.5:** 55–65% (small objects)
- **mAP@0.5:0.95:** 29.97%  

> 🔍 *mAP@0.5:0.95 measures detection consistency across IoU thresholds. Scores are competitive for lightweight models and will improve with further tuning.*

---
