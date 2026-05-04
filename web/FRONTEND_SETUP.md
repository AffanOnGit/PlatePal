# PlatePal Frontend: Quick Start Guide

This directory contains the premium React-based UI for PlatePal. It is built using **Vite** for maximum performance and **Axios** for seamless communication with the GenAI backend.

## 🛠️ Prerequisites
- **Node.js** (v18 or higher recommended)
- **npm** (comes bundled with Node.js)

## 🚀 Installation on a New Machine
If you are running this frontend for the first time or on a different machine, follow these steps:

1. **Navigate to the web directory**:
   ```bash
   cd web
   ```

2. **Install Dependencies**:
   This will read the `package.json` and download all necessary libraries (React, Axios, Vite, etc.):
   ```bash
   npm install
   ```

3. **Start the Development Server**:
   ```bash
   npm run dev
   ```

4. **Access the App**:
   Open your browser and go to: `http://localhost:5173`

## 🔗 Backend Connectivity
The frontend is configured to communicate with the FastAPI backend. Ensure your backend is running at `http://localhost:8000` (or update the API URL in `src/App.jsx` if using a tunnel).

## 📦 Key Dependencies
- **React 19**: Modern UI framework.
- **Axios**: Handling asynchronous API requests to the GenAI engine.
- **Vite 6**: Next-generation frontend tooling for instant HMR.
- **Lucide React**: Premium iconography.
