import React, { useState } from 'react';
import { FileText, Image, TrendingUp, Code, BookOpen, CheckCircle } from 'lucide-react';

const PricingSolution = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BookOpen },
    { id: 'approach', label: 'Approach', icon: TrendingUp },
    { id: 'code', label: 'Implementation', icon: Code },
    { id: 'features', label: 'Features', icon: FileText }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-8 text-white">
            <h1 className="text-4xl font-bold mb-2">Smart Product Pricing Challenge</h1>
            <p className="text-blue-100">Multimodal ML Solution for E-commerce Price Prediction</p>
          </div>

          {/* Tabs */}
          <div className="flex border-b bg-gray-50">
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'border-b-2 border-blue-600 text-blue-600 bg-white'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon size={18} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Content */}
          <div className="p-8">
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <div className="bg-blue-50 border-l-4 border-blue-600 p-4 rounded">
                  <h2 className="text-2xl font-bold text-blue-900 mb-2">Challenge Overview</h2>
                  <p className="text-gray-700">
                    Predict product prices using multimodal features (text + images) with SMAPE evaluation metric
                  </p>
                </div>

                <div className="grid md:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200">
                    <div className="text-green-600 mb-2">
                      <FileText size={32} />
                    </div>
                    <h3 className="font-bold text-lg mb-2">Training Data</h3>
                    <p className="text-gray-700">75,000 products with catalog text, images, and prices</p>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg border border-purple-200">
                    <div className="text-purple-600 mb-2">
                      <Image size={32} />
                    </div>
                    <h3 className="font-bold text-lg mb-2">Multimodal</h3>
                    <p className="text-gray-700">Text descriptions + Product images for richer features</p>
                  </div>

                  <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-lg border border-orange-200">
                    <div className="text-orange-600 mb-2">
                      <TrendingUp size={32} />
                    </div>
                    <h3 className="font-bold text-lg mb-2">Evaluation</h3>
                    <p className="text-gray-700">SMAPE metric on 75K test samples</p>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                  <h3 className="font-bold text-yellow-900 mb-2">⚠️ Important Constraints</h3>
                  <ul className="list-disc list-inside text-gray-700 space-y-1">
                    <li>NO external price lookup allowed</li>
                    <li>Model must be MIT/Apache 2.0 licensed, ≤8B parameters</li>
                    <li>All predictions must be positive floats</li>
                  </ul>
                </div>
              </div>
            )}

            {activeTab === 'approach' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Solution Strategy</h2>
                
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
                  <h3 className="font-bold text-xl mb-3 text-indigo-900">Ensemble Approach</h3>
                  <p className="text-gray-700 mb-4">
                    Combining multiple models for robust predictions across different price ranges
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white border-2 border-gray-200 p-6 rounded-lg">
                    <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                      <FileText className="text-blue-600" size={24} />
                      Text Features
                    </h3>
                    <ul className="space-y-2 text-gray-700">
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>TF-IDF vectorization for text content</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Extract brand, quantity, specifications</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Text length, word count statistics</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Sentence embeddings for semantic features</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-white border-2 border-gray-200 p-6 rounded-lg">
                    <h3 className="font-bold text-lg mb-3 flex items-center gap-2">
                      <Image className="text-purple-600" size={24} />
                      Image Features
                    </h3>
                    <ul className="space-y-2 text-gray-700">
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Pre-trained CNN embeddings (ResNet/EfficientNet)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Color histograms and dominant colors</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Image quality metrics</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle size={18} className="text-green-500 mt-1 flex-shrink-0" />
                        <span>Texture and edge features</span>
                      </li>
                    </ul>
                  </div>
                </div>

                <div className="bg-indigo-50 border-2 border-indigo-200 p-6 rounded-lg">
                  <h3 className="font-bold text-lg mb-3 text-indigo-900">Model Ensemble</h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="bg-white p-4 rounded border border-indigo-200">
                      <h4 className="font-bold mb-2">XGBoost</h4>
                      <p className="text-sm text-gray-600">Gradient boosting for complex patterns</p>
                    </div>
                    <div className="bg-white p-4 rounded border border-indigo-200">
                      <h4 className="font-bold mb-2">LightGBM</h4>
                      <p className="text-sm text-gray-600">Fast, efficient tree-based model</p>
                    </div>
                    <div className="bg-white p-4 rounded border border-indigo-200">
                      <h4 className="font-bold mb-2">CatBoost</h4>
                      <p className="text-sm text-gray-600">Handles categorical features well</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'code' && (
              <div className="space-y-4">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Implementation Pipeline</h2>
                
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                  <div className="mb-2 text-gray-400"># Main Pipeline Steps:</div>
                  <div>1. Load and preprocess data</div>
                  <div>2. Extract text features (TF-IDF, embeddings)</div>
                  <div>3. Download and process images</div>
                  <div>4. Extract image features (CNN embeddings)</div>
                  <div>5. Combine features and train ensemble</div>
                  <div>6. Generate predictions with proper formatting</div>
                </div>

                <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                  <h3 className="font-bold mb-2">Key Libraries Required:</h3>
                  <div className="grid md:grid-cols-2 gap-2 text-sm">
                    <div className="font-mono bg-white p-2 rounded">pandas, numpy, scikit-learn</div>
                    <div className="font-mono bg-white p-2 rounded">xgboost, lightgbm, catboost</div>
                    <div className="font-mono bg-white p-2 rounded">torch, torchvision, timm</div>
                    <div className="font-mono bg-white p-2 rounded">sentence-transformers, Pillow</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h3 className="font-bold text-lg">Complete Implementation Available</h3>
                  <p className="text-gray-700">
                    I'll provide the full Python implementation including:
                  </p>
                  <ul className="list-disc list-inside text-gray-700 space-y-1 ml-4">
                    <li>Feature extraction pipeline</li>
                    <li>Model training with cross-validation</li>
                    <li>Ensemble prediction with optimal weights</li>
                    <li>Output formatting matching requirements</li>
                  </ul>
                </div>
              </div>
            )}

            {activeTab === 'features' && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">Feature Engineering Details</h2>
                
                <div className="space-y-4">
                  <div className="border-l-4 border-blue-500 pl-4 py-2">
                    <h3 className="font-bold text-lg mb-2">Text Processing Features</h3>
                    <ul className="space-y-2 text-gray-700">
                      <li><strong>TF-IDF Features:</strong> Top 1000-5000 n-grams (unigrams, bigrams)</li>
                      <li><strong>Statistical Features:</strong> Text length, word count, unique words, avg word length</li>
                      <li><strong>Price Indicators:</strong> Presence of pack quantities, numbers in text</li>
                      <li><strong>Brand Extraction:</strong> Identify brand names from common patterns</li>
                      <li><strong>Semantic Embeddings:</strong> Sentence-BERT for semantic meaning (384-dim)</li>
                    </ul>
                  </div>

                  <div className="border-l-4 border-purple-500 pl-4 py-2">
                    <h3 className="font-bold text-lg mb-2">Image Processing Features</h3>
                    <ul className="space-y-2 text-gray-700">
                      <li><strong>CNN Embeddings:</strong> EfficientNet-B0 or ResNet18 features (512-2048 dim)</li>
                      <li><strong>Color Features:</strong> RGB histograms, dominant colors, color variance</li>
                      <li><strong>Basic Features:</strong> Image dimensions, aspect ratio, file size indicators</li>
                      <li><strong>Quality Metrics:</strong> Brightness, contrast, sharpness estimates</li>
                    </ul>
                  </div>

                  <div className="border-l-4 border-green-500 pl-4 py-2">
                    <h3 className="font-bold text-lg mb-2">Engineered Features</h3>
                    <ul className="space-y-2 text-gray-700">
                      <li><strong>Price Range Indicators:</strong> Keywords suggesting price ranges</li>
                      <li><strong>Product Categories:</strong> Inferred from text patterns</li>
                      <li><strong>Quality Signals:</strong> Premium keywords, professional descriptions</li>
                      <li><strong>Interaction Features:</strong> Combinations of text and image features</li>
                    </ul>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg border-2 border-green-300">
                  <h3 className="font-bold text-lg mb-2 text-green-900">Expected Performance</h3>
                  <p className="text-gray-700 mb-2">
                    With proper feature engineering and ensemble methods:
                  </p>
                  <ul className="list-disc list-inside text-gray-700 space-y-1">
                    <li>Target SMAPE: &lt;15-20% on validation set</li>
                    <li>Robust across different price ranges</li>
                    <li>Handles missing images gracefully</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 text-center text-gray-600 text-sm">
          <p>Ready to implement the complete solution? Let me know if you'd like the full Python code!</p>
        </div>
      </div>
    </div>
  );
};

export default PricingSolution;