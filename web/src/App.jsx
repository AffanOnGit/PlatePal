import React, { useState } from 'react';
import axios from 'axios';

/**
 * PlatePal — Premium Generative AI Plating UI
 * ===========================================
 * Designed for a high-end academic showcase.
 * Features: High-contrast dark mode, serif typography, 
 * and AI-enhanced plating visualization.
 */

function App() {
  const [ingredients, setIngredients] = useState('');
  const [recipe, setRecipe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!ingredients.trim()) return;
    
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/generate', {
        ingredients: ingredients
      });
      setRecipe(response.data);
    } catch (err) {
      console.error(err);
      setError('The AI chef is currently resting. Please try again in a moment.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="platepal-app">
      {/* --- HERO SECTION --- */}
      <header className="hero">
        <h1 className="logo">PlatePal</h1>
        <p className="subtitle">AI-Powered Gastronomy & Visual Plating</p>
        
        <div className="input-container">
          <input
            type="text"
            value={ingredients}
            onChange={(e) => setIngredients(e.target.value)}
            placeholder="e.g., scallops, saffron, fennel, cream"
            onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
          />
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? 'Imagining...' : 'Generate Concept'}
          </button>
        </div>
      </header>

      {/* --- RESULTS SECTION --- */}
      <main className="results">
        {error && <div className="error-card">{error}</div>}

        {recipe && (
          <div className="showcase-grid">
            
            {/* LEFT: Recipe Info */}
            <div className="recipe-panel">
              <h2 className="recipe-title">{recipe.recipe.title}</h2>
              
              <div className="section">
                <h3>Ingredients</h3>
                <div className="ingredient-list">
                  {recipe.recipe.ingredients.split(',').map((ing, i) => (
                    <span key={i} className="ing-tag">{ing.trim()}</span>
                  ))}
                </div>
              </div>

              <div className="section">
                <h3>Instructions</h3>
                <p className="instructions-text">
                  {recipe.recipe.instructions}
                </p>
              </div>
            </div>

            {/* RIGHT: Visual Plating */}
            <div className="visual-panel">
              <div className="plate-container">
                <img 
                  src={`data:image/png;base64,${recipe.image_base64}`} 
                  alt="AI Plating Visual" 
                />
                <div className="plate-overlay"></div>
              </div>
              <div className="visual-meta">
                <span>Plating Concept: {recipe.metadata?.image_model || 'AI Gen'}</span>
                <span>Resolution: {recipe.metadata?.resolution || '512px'} (AI Enhanced)</span>
              </div>
            </div>

          </div>
        )}

        {!recipe && !loading && (
          <div className="placeholder-state">
            <p>Enter your ingredients to visualize a professional dish concept.</p>
          </div>
        )}
      </main>

      {/* --- STYLES --- */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;600&display=swap');

        :root {
          --bg-dark: #0a0a0a;
          --card-bg: #141414;
          --accent: #d4af37; /* Gold */
          --text-main: #f0f0f0;
          --text-dim: #a0a0a0;
          --serif: 'Playfair Display', serif;
          --sans: 'Inter', sans-serif;
        }

        .platepal-app {
          min-height: 100vh;
          background: var(--bg-dark);
          color: var(--text-main);
          font-family: var(--sans);
          padding: 2rem;
        }

        /* Hero */
        .hero {
          text-align: center;
          margin-bottom: 4rem;
        }
        .logo {
          font-family: var(--serif);
          font-size: 3.5rem;
          margin-bottom: 0.5rem;
          background: linear-gradient(to right, #fff, var(--accent));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .subtitle {
          color: var(--text-dim);
          letter-spacing: 2px;
          text-transform: uppercase;
          font-size: 0.8rem;
          margin-bottom: 2rem;
        }

        /* Input */
        .input-container {
          max-width: 600px;
          margin: 0 auto;
          display: flex;
          gap: 1rem;
          background: #1a1a1a;
          padding: 0.5rem;
          border-radius: 50px;
          border: 1px solid #333;
        }
        input {
          flex: 1;
          background: transparent;
          border: none;
          color: white;
          padding: 0 1.5rem;
          font-size: 1rem;
        }
        input:focus { outline: none; }
        button {
          background: var(--accent);
          color: black;
          border: none;
          padding: 0.8rem 2rem;
          border-radius: 50px;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }

        /* Showcase Grid */
        .showcase-grid {
          display: flex;
          flex-direction: row;
          align-items: flex-start;
          gap: 2rem;
          max-width: 1200px;
          margin: 0 auto;
          animation: fadeIn 0.8s ease-out;
        }

        .recipe-panel {
          flex: 1;
          background: var(--card-bg);
          padding: 2.5rem;
          border-radius: 24px;
          border: 1px solid #222;
          min-width: 0; /* Prevents overflow */
        }
        .recipe-title {
          font-family: var(--serif);
          font-size: 2.8rem;
          line-height: 1.1;
          margin-bottom: 2rem;
        }
        .section h3 {
          text-transform: uppercase;
          font-size: 0.75rem;
          letter-spacing: 1px;
          color: var(--accent);
          margin-bottom: 1rem;
        }
        .ingredient-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 2rem;
        }
        .ing-tag {
          background: #222;
          padding: 0.4rem 1rem;
          border-radius: 4px;
          font-size: 0.9rem;
          color: var(--text-dim);
        }
        .instructions-text {
          color: var(--text-dim);
          line-height: 1.6;
          white-space: pre-wrap;
        }

        /* Visual Panel */
        .visual-panel {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          min-width: 400px;
        }
        .plate-container {
          position: relative;
          aspect-ratio: 1/1;
          background: #000;
          border-radius: 24px;
          overflow: hidden;
          border: 1px solid #333;
          box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        }
        .plate-container img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }
        .plate-overlay {
          position: absolute;
          inset: 0;
          box-shadow: inset 0 0 100px rgba(0,0,0,0.8);
          pointer-events: none;
        }
        .visual-meta {
          display: flex;
          justify-content: space-between;
          font-size: 0.7rem;
          color: #444;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .placeholder-state {
          text-align: center;
          color: #444;
          margin-top: 4rem;
        }
      `}</style>
    </div>
  );
}

export default App;
