import React from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import AnalyzerPage from "./pages/AnalyzerPage.jsx";
import BatchPage from "./pages/BatchPage.jsx";
import ModelInfoPage from "./pages/ModelInfoPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import AboutPage from "./pages/AboutPage.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<AnalyzerPage />} />
        <Route path="/batch" element={<BatchPage />} />
        <Route path="/model-info" element={<ModelInfoPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  );
}
