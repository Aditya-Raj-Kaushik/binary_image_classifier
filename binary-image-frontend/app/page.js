'use client';

import { useState } from 'react';

export default function HomePage() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append("file", image);

    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Upload failed:", error);
      setResult({ prediction: "Error", confidence: 0 });
    }
    setLoading(false);
  };

  const handleDiscard = () => {
    setImage(null);
    setResult(null);
    document.getElementById('image-input').value = null; 
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4">Fruit Classifier</h1>

      <input
        id="image-input"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
      />

      <div className="mt-4 flex space-x-4">
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded"
          onClick={handleUpload}
          disabled={!image || loading}
        >
          Predict
        </button>
        <button
          className="px-4 py-2 bg-red-500 text-white rounded"
          onClick={handleDiscard}
          disabled={!image && !result}
        >
          Discard
        </button>
      </div>

      {loading && <p className="mt-4">Processing...</p>}

      {result && (
        <div className="mt-4">
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Confidence:</strong> {result.confidence}</p>
        </div>
      )}
    </div>
  );
}
