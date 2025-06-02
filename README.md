# 🧠 Maze Game Using Hand Gestures

This project implements a gesture-controlled game using machine learning for gesture recognition. Players can control the game using hand gestures captured through their webcam.

## 🚀 Quick Start

1. Install the Live Server extension in VS Code:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Live Server"
   - Install the extension by Ritwick Dey

2. Launch the project:
   - Right-click on `index.html`
   - Select "Open with Live Server"
   - The game should open in your default browser at `http://localhost:5500`

## 📁 Project Structure

- `index.html` - Main game interface
- `api-call.js` - ML model API integration
- `cam.js` - Webcam handling and gesture processing
- `keyboard.js` - Keyboard controls implementation
- `maze.js` - Maze game logic
- `mp.js` - Media processing utilities

## 🎮 Controls

You can control the game using:

### ✋ Hand Gestures (via Webcam)

✌️ Peace sign → Move Left

☝️ One finger → Move Right

👍 Thumbs up → Move Up

👎 Thumbs down → Move Down


### ⌨️ Keyboard (Fallback Controls)

Arrow keys → Move in respective directions
