import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io
import base64
import os
import shap
import networkx as nx
import matplotlib.patches as mpatches

# 设置中文字体 - 放在最前面
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

# 设置页面配置
st.set_page_config(page_title="烧伤智能识别系统", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# 自定义CSS样式
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #ff6b35; text-align: center; margin-bottom: 2rem; font-weight: bold; font-family: "Microsoft YaHei", sans-serif; }
    .sub-header { font-size: 1.5rem; color: #ff8e53; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .feature-box { background-color: #fff5f5; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff6b35; margin: 0.5rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .prediction-box { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .analysis-box { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #2196F3; margin: 1rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .setting-box { background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #627d98; margin: 0.5rem 0; font-family: "Microsoft YaHei", sans-serif; }
    .guide-section { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #6c757d; font-family: "Microsoft YaHei", sans-serif; }
    .theory-box { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #ffc107; font-family: "Microsoft YaHei", sans-serif; }
    .code-box { background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #6c757d; font-family: "Courier New", monospace; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# 加载预训练模型
@st.cache_resource
def load_model():
    try:
        model_path = "rf.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if not hasattr(model, 'feature_names_in_'): model.feature_names_in_ = ['BG1', 'EGF', 'IL-1β', 'BG2']
            return model
        else:
            st.error(f"❌ 模型文件未找到: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

# 获取图表字体设置函数
def get_chart_font_settings():
    """获取图表字体设置"""
    return {
        'title_font': st.session_state.get('chart_title_font', {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}),
        'axis_font': st.session_state.get('chart_axis_font', {'family': 'Microsoft YaHei', 'size': 10}),
        'tick_font': st.session_state.get('chart_tick_font', {'family': 'Microsoft YaHei', 'size': 8}),
        'label_font': st.session_state.get('chart_label_font', {'family': 'Microsoft YaHei', 'size': 9})
    }

# 应用图表字体设置函数
def apply_chart_font_settings(ax=None, title=None, xlabel=None, ylabel=None):
    """应用图表字体设置"""
    font_settings = get_chart_font_settings()
    
    if ax is not None:
        # 设置标题字体
        if title and ax.get_title():
            ax.set_title(ax.get_title(), fontfamily=font_settings['title_font']['family'], 
                        fontsize=font_settings['title_font']['size'], fontweight=font_settings['title_font']['weight'])
        
        # 设置坐标轴标签字体
        if xlabel or ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel() if not xlabel else xlabel, 
                         fontfamily=font_settings['axis_font']['family'], 
                         fontsize=font_settings['axis_font']['size'])
        
        if ylabel or ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel() if not ylabel else ylabel, 
                         fontfamily=font_settings['axis_font']['family'], 
                         fontsize=font_settings['axis_font']['size'])
        
        # 设置刻度标签字体
        ax.tick_params(axis='both', which='major', 
                      labelsize=font_settings['tick_font']['size'])
        
        # 设置图例字体（如果存在）
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontfamily(font_settings['label_font']['family'])
                text.set_fontsize(font_settings['label_font']['size'])

# SHAP分析函数
def perform_shap_analysis(model, input_data, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        prediction = model.predict(input_data)[0]
        
        if shap_values.ndim == 3:
            current_shap_values = shap_values[0, :, prediction]
        else:
            st.error(f"不支持的SHAP维度: {shap_values.ndim}")
            return None
        
        if current_shap_values.ndim > 1: current_shap_values = current_shap_values[0]
        
        feature_importance = np.abs(current_shap_values)
        sorted_idx = np.argsort(feature_importance)[::-1]
        
        return {
            'shap_values': current_shap_values, 'shap_values_3d': shap_values, 'input_data': input_data,
            'feature_importance': feature_importance, 'sorted_features': [feature_names[i] for i in sorted_idx],
            'sorted_importance': feature_importance[sorted_idx], 'prediction': prediction
        }
    except Exception as e:
        st.error(f"SHAP分析错误: {str(e)}")
        return None

# 图1: 合并的SHAP分析图表
def plot_combined_shap_analysis(shap_results, feature_names, burn_type_mapping):
    try:
        if shap_results is None: return None
        shap_values_3d = shap_results['shap_values_3d']
        prediction = shap_results['prediction']
        
        # 获取字体设置
        font_settings = get_chart_font_settings()
        
        # 设置全局字体
        plt.rcParams.update({
            'font.size': font_settings['tick_font']['size'],
            'axes.titlesize': font_settings['title_font']['size'],
            'axes.labelsize': font_settings['axis_font']['size'],
            'xtick.labelsize': font_settings['tick_font']['size'],
            'ytick.labelsize': font_settings['tick_font']['size'],
            'font.family': font_settings['title_font']['family']
        })
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SHAP Analysis: Feature Impact and Importance for All Classes', 
                     fontsize=font_settings['title_font']['size'] + 2, 
                     fontweight='bold', y=0.95,
                     fontfamily=font_settings['title_font']['family'])
        
        for i in range(6):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if shap_values_3d.ndim == 3:
                class_shap_values = np.mean(shap_values_3d[:, :, i], axis=0)
                class_shap_importance = np.mean(np.abs(shap_values_3d[:, :, i]), axis=0)
            else:
                class_shap_values = shap_values_3d[i]
                class_shap_importance = np.abs(shap_values_3d[i])
            
            sorted_idx = np.argsort(class_shap_importance)[::-1]
            sorted_features = [feature_names[j] for j in sorted_idx]
            sorted_shap = class_shap_values[sorted_idx]
            sorted_importance = class_shap_importance[sorted_idx]
            
            y_pos = np.arange(len(sorted_features))
            colors = ['#ff6b6b' if val > 0 else '#4ecdc4' for val in sorted_shap]
            bars = ax.barh(y_pos, sorted_shap, color=colors, alpha=0.8, height=0.6)
            
            for j, (shap_val, imp_val) in enumerate(zip(sorted_shap, sorted_importance)):
                ax.scatter(imp_val if shap_val >= 0 else -imp_val, j, s=80, color='#2d3436', marker='o', alpha=0.7, zorder=5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontfamily=font_settings['tick_font']['family'])
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
            ax.set_xlabel('SHAP Value / Importance', 
                         fontsize=font_settings['axis_font']['size'], 
                         fontweight='bold',
                         fontfamily=font_settings['axis_font']['family'])
            ax.grid(True, alpha=0.3, axis='x')
            
            if i == prediction:
                ax.patch.set_facecolor('#fffacd')
                ax.patch.set_alpha(0.3)
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                title_color = 'red'
                title_suffix = ' ★'
            else:
                title_color = 'black'
                title_suffix = ''
            
            ax.set_title(f'Class {i}: {burn_type_mapping[i]["en"]}{title_suffix}', 
                        fontsize=font_settings['title_font']['size'], 
                        fontweight='bold', color=title_color, pad=10,
                        fontfamily=font_settings['title_font']['family'])
            
            for j, (bar, shap_val, imp_val) in enumerate(zip(bars, sorted_shap, sorted_importance)):
                width = bar.get_width()
                if abs(shap_val) > 0.001:
                    if shap_val > 0:
                        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2., f'{shap_val:+.6f}', 
                               ha='left', va='center', 
                               fontsize=font_settings['label_font']['size'] - 1, 
                               color='#d63031', fontweight='bold',
                               fontfamily=font_settings['label_font']['family'])
                    else:
                        ax.text(width - 0.005, bar.get_y() + bar.get_height()/2., f'{shap_val:+.6f}', 
                               ha='right', va='center', 
                               fontsize=font_settings['label_font']['size'] - 1, 
                               color='#00b894', fontweight='bold',
                               fontfamily=font_settings['label_font']['family'])
                    
                    ax.text(imp_val + 0.005 if shap_val >= 0 else -imp_val - 0.005, j, f'{imp_val:.6f}', 
                           ha='left' if shap_val >= 0 else 'right', va='center', 
                           fontsize=font_settings['label_font']['size'] - 2, 
                           color='#2d3436', fontweight='bold',
                           fontfamily=font_settings['label_font']['family'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff6b6b', alpha=0.8, label='Positive Impact'),
            Patch(facecolor='#4ecdc4', alpha=0.8, label='Negative Impact'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2d3436', markersize=6, label='Importance Magnitude')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, 
                  fontsize=font_settings['label_font']['size'], framealpha=0.9, fancybox=True, shadow=True)
        
        return fig
    except Exception as e:
        st.error(f"SHAP图表绘制错误: {str(e)}")
        return None

# 图2: 当前预测类别的特征重要性图
def plot_current_prediction_shap(shap_results, feature_names, burn_type_mapping):
    try:
        if shap_results is None: return None
        prediction = shap_results['prediction']
        sorted_features = shap_results['sorted_features']
        sorted_importance = shap_results['sorted_importance']
        
        # 获取字体设置
        font_settings = get_chart_font_settings()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'SHAP Analysis for Current Prediction: {burn_type_mapping[prediction]["en"]}', 
                     fontsize=font_settings['title_font']['size'] + 2, fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        
        # 左侧：特征重要性条形图
        y_pos = np.arange(len(sorted_features))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
        bars = ax1.barh(y_pos, sorted_importance, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_features, fontfamily=font_settings['tick_font']['family'])
        ax1.invert_yaxis()
        ax1.set_xlabel('SHAP Value Importance', fontweight='bold',
                       fontfamily=font_settings['axis_font']['family'],
                       fontsize=font_settings['axis_font']['size'])
        ax1.set_title('Feature Importance Ranking', fontweight='bold',
                     fontfamily=font_settings['title_font']['family'],
                     fontsize=font_settings['title_font']['size'])
        ax1.grid(True, alpha=0.3, axis='x')
        
        for bar, importance in zip(bars, sorted_importance):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2., f'{width:.10f}', 
                    ha='left', va='center', 
                    fontsize=font_settings['label_font']['size'], fontweight='bold',
                    fontfamily=font_settings['label_font']['family'])
        
        # 右侧：SHAP值正负影响饼图
        shap_values = shap_results['shap_values']
        positive_count = np.sum(shap_values > 0)
        negative_count = np.sum(shap_values < 0)
        neutral_count = np.sum(shap_values == 0)
        
        sizes = [positive_count, negative_count, neutral_count]
        labels = ['Positive Impact', 'Negative Impact', 'No Impact']
        colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                                             textprops={'fontfamily': font_settings['label_font']['family'],
                                                       'fontsize': font_settings['label_font']['size']})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'No significant\nSHAP values', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=font_settings['label_font']['size'],
                    fontfamily=font_settings['label_font']['family'])
        
        ax2.set_title('SHAP Value Distribution', fontweight='bold',
                     fontfamily=font_settings['title_font']['family'],
                     fontsize=font_settings['title_font']['size'])
        
        # 应用字体设置
        apply_chart_font_settings(ax1, xlabel='SHAP Value Importance')
        apply_chart_font_settings(ax2)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"当前预测SHAP图表绘制错误: {str(e)}")
        return None

# 优化的图网络分析
def perform_graph_analysis(feature_values, feature_names, prediction, burn_type_mapping):
    try:
        G = nx.Graph()
        for i, feature in enumerate(feature_names):
            G.add_node(feature, value=feature_values[i], importance=abs(feature_values[i]))
        
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                correlation = 1 - abs(feature_values[i] - feature_values[j]) / (abs(feature_values[i]) + abs(feature_values[j]) + 1e-8)
                if correlation > 0.3:
                    G.add_edge(feature_names[i], feature_names[j], weight=correlation)
        
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        return {
            'graph': G, 'degree_centrality': degree_centrality, 'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality, 'node_importance': {feature: abs(val) for feature, val in zip(feature_names, feature_values)}
        }
    except Exception as e:
        st.warning(f"图网络分析遇到问题: {str(e)}")
        return None

# 优化的图网络可视化
def plot_optimized_graph_analysis(graph_results, feature_names, burn_info):
    try:
        if graph_results is None: return None
        G = graph_results['graph']
        
        # 获取字体设置
        font_settings = get_chart_font_settings()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Feature Network Analysis - {burn_info["cn"]}', 
                     fontsize=font_settings['title_font']['size'] + 2, fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        
        # 图1: 网络拓扑图
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')
        
        pos = nx.spring_layout(G, seed=42, k=3, iterations=200)
        
        node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        node_color_map = {feature: node_colors[i] for i, feature in enumerate(feature_names)}
        
        node_sizes = [3000 + 2000 * graph_results['node_importance'][node] for node in G.nodes()]
        node_colors_list = [node_color_map[node] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors_list, 
                              alpha=0.9, ax=ax1, edgecolors='black', linewidths=2)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        edge_colors = ['#2C3E50' for _ in edges]
        edge_widths = [w * 5 + 1 for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=[min(w * 1.5, 0.8) for w in weights],
                              edge_color=edge_colors, ax=ax1, style='solid')
        
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, 
                              font_weight='bold', ax=ax1,
                              font_family=font_settings['label_font']['family'],
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        
        ax1.set_title('Network Topology', fontsize=font_settings['title_font']['size'], fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        ax1.axis('off')
        
        # 图2: 中心性分析雷达图
        centrality_data = {
            'Feature': list(graph_results['degree_centrality'].keys()),
            'Degree': list(graph_results['degree_centrality'].values()),
            'Betweenness': list(graph_results['betweenness_centrality'].values()),
            'Closeness': list(graph_results['closeness_centrality'].values())
        }
        df = pd.DataFrame(centrality_data)
        
        categories = list(df['Feature'])
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax2 = plt.subplot(132, polar=True)
        ax2.set_facecolor('white')
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontfamily=font_settings['tick_font']['family'])
        
        values = df['Degree'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Degree Centrality', color='#e74c3c')
        ax2.fill(angles, values, alpha=0.25, color='#e74c3c')
        
        values = df['Betweenness'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Betweenness Centrality', color='#3498db')
        ax2.fill(angles, values, alpha=0.25, color='#3498db')
        
        values = df['Closeness'].values.tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label='Closeness Centrality', color='#2ecc71')
        ax2.fill(angles, values, alpha=0.25, color='#2ecc71')
        
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                  prop={'family': font_settings['label_font']['family'], 'size': font_settings['label_font']['size']})
        ax2.set_title('Centrality Analysis Radar Chart', fontsize=font_settings['title_font']['size'], fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        
        # 图3: 特征关联热力图
        ax3.set_facecolor('white')
        correlation_matrix = np.zeros((len(feature_names), len(feature_names)))
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if feat1 == feat2:
                    correlation_matrix[i, j] = 1.0
                elif G.has_edge(feat1, feat2):
                    correlation_matrix[i, j] = G[feat1][feat2]['weight']
                else:
                    correlation_matrix[i, j] = 0.0
        
        im = ax3.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax3.set_xticks(range(len(feature_names)))
        ax3.set_yticks(range(len(feature_names)))
        ax3.set_xticklabels(feature_names, rotation=45, fontfamily=font_settings['tick_font']['family'])
        ax3.set_yticklabels(feature_names, fontfamily=font_settings['tick_font']['family'])
        ax3.set_title('Feature Correlation Heatmap', fontsize=font_settings['title_font']['size'], fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.6f}', ha="center", va="center", color="black", 
                              fontsize=font_settings['label_font']['size'] - 1, fontweight='bold',
                              fontfamily=font_settings['label_font']['family'])
        
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        apply_chart_font_settings(ax1)
        apply_chart_font_settings(ax2)
        apply_chart_font_settings(ax3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"图网络可视化错误: {str(e)}")
        return None

# 计算综合特征权重
def calculate_integrated_feature_weights(shap_results, graph_results, feature_names, alpha=0.4, beta=0.2, gamma=0.2, delta=0.2):
    """
    计算综合特征权重: w_i = α·normalize(|ϕᵢ|) + β·normalize(DCᵢ) + γ·normalize(BCᵢ) + δ·normalize(CCᵢ)
    """
    try:
        # 1. SHAP重要性归一化
        shap_values = shap_results['shap_values']
        shap_importance = np.abs(shap_values)
        shap_norm = (shap_importance - np.min(shap_importance)) / (np.max(shap_importance) - np.min(shap_importance) + 1e-8)
        
        # 2. 图网络中心性归一化
        dc_values = np.array([graph_results['degree_centrality'][f] for f in feature_names])
        bc_values = np.array([graph_results['betweenness_centrality'][f] for f in feature_names])
        cc_values = np.array([graph_results['closeness_centrality'][f] for f in feature_names])
        
        dc_norm = (dc_values - np.min(dc_values)) / (np.max(dc_values) - np.min(dc_values) + 1e-8)
        bc_norm = (bc_values - np.min(bc_values)) / (np.max(bc_values) - np.min(bc_values) + 1e-8)
        cc_norm = (cc_values - np.min(cc_values)) / (np.max(cc_values) - np.min(cc_values) + 1e-8)
        
        # 3. 计算综合权重
        integrated_weights = alpha * shap_norm + beta * dc_norm + gamma * bc_norm + delta * cc_norm
        
        # 4. 归一化
        integrated_weights = integrated_weights / np.sum(integrated_weights)
        
        return dict(zip(feature_names, integrated_weights))
    
    except Exception as e:
        st.warning(f"计算综合特征权重错误: {str(e)}")
        return {f: 1.0/len(feature_names) for f in feature_names}  # 默认均匀分布

# 烧伤严重程度定义
BURN_SEVERITY_ORDER = [0, 1, 2, 3, 4, 5]  # 从最轻到最重

def get_target_priorities(current_class):
    """获取目标类别优先级（从最优先到最不优先）"""
    if current_class not in BURN_SEVERITY_ORDER:
        return []
    
    current_idx = BURN_SEVERITY_ORDER.index(current_class)
    
    # 1. 直接更轻度烧伤（最高优先级）
    if current_idx > 0:
        direct_milder = [BURN_SEVERITY_ORDER[current_idx - 1]]
    else:
        direct_milder = []
    
    # 2. 多级降级
    milder_multi = []
    for i in range(current_idx - 2, -1, -1):
        if i >= 0:
            milder_multi.append(BURN_SEVERITY_ORDER[i])
    
    # 3. 正常组织（特殊目标）
    normal_tissue = [0] if 0 not in direct_milder + milder_multi else []
    
    # 4. 其他类别
    other_classes = []
    for cls in BURN_SEVERITY_ORDER:
        if cls != current_class and cls not in direct_milder + milder_multi + normal_tissue:
            other_classes.append(cls)
    
    # 合并所有优先级
    all_targets = direct_milder + milder_multi + normal_tissue + other_classes
    
    return all_targets

def find_path_to_target(model, base_values, original_class, target_class, 
                        feature_names, shap_values, max_attempts=20):
    """寻找从当前类别到目标类别的可行路径"""
    suggestions = []
    n_features = len(feature_names)
    
    # 1. 单特征修改
    for attempt in range(min(max_attempts, 20)):
        for i in range(n_features):
            shap_dir = shap_values[i]
            
            # 确定修改方向
            if shap_dir > 0:
                change_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
            elif shap_dir < 0:
                change_factors = [1.1, 1.3, 1.5, 2.0, 3.0]
            else:
                change_factors = [0.5, 0.7, 1.3, 1.5, 2.0]
            
            for factor in change_factors:
                modified_data = base_values.copy()
                modified_data[i] = base_values[i] * factor
                
                modified_df = pd.DataFrame([modified_data], columns=feature_names)
                new_prediction = model.predict(modified_df)[0]
                new_probability = model.predict_proba(modified_df)[0][new_prediction]
                
                if new_prediction == target_class:
                    suggestions.append({
                        'feature': feature_names[i],
                        'change_factor': factor,
                        'confidence': new_probability,
                        'original_value': base_values[i],
                        'new_value': modified_data[i],
                        'direction': '减少' if factor < 1 else '增加',
                        'target_class': target_class
                    })
                    
                    if len(suggestions) >= 3:
                        return suggestions
    
    return suggestions

def calculate_improvement(from_class, to_class):
    """计算改善程度"""
    if from_class not in BURN_SEVERITY_ORDER or to_class not in BURN_SEVERITY_ORDER:
        return 0
    
    from_idx = BURN_SEVERITY_ORDER.index(from_class)
    to_idx = BURN_SEVERITY_ORDER.index(to_class)
    
    if to_idx < from_idx:  # 改善
        if to_class == 0:  # 变为正常组织
            return 100
        improvement = (from_idx - to_idx) * 20
        return min(improvement, 80)
    elif to_idx > from_idx:  # 恶化
        return -20
    else:  # 不变
        return 0

# 反事实分析函数 - 添加降级处理
def perform_counterfactual_analysis(model, input_data, original_prediction, feature_names, burn_type_mapping, shap_results=None, graph_results=None):
    try:
        base_values = input_data.iloc[0].values
        shap_values = shap_results['shap_values'] if shap_results else np.zeros(len(feature_names))
        
        # 计算特征权重
        feature_weights = {}
        if shap_results and graph_results:
            feature_weights = calculate_integrated_feature_weights(shap_results, graph_results, feature_names)
        else:
            feature_weights = {f: 1.0/len(feature_names) for f in feature_names}
        
        all_suggestions = []
        normal_tissue_suggestions = []
        milder_suggestions = []
        other_suggestions = []
        
        # 获取目标优先级
        target_priorities = get_target_priorities(original_prediction)
        
        # 按优先级搜索可行路径
        for target_class in target_priorities:
            suggestions = find_path_to_target(
                model, base_values, original_prediction, target_class,
                feature_names, shap_values, max_attempts=20
            )
            
            for sug in suggestions:
                sug['target_class'] = target_class
                sug['target_name'] = burn_type_mapping[target_class]['cn']
                sug['improvement'] = calculate_improvement(original_prediction, target_class)
                sug['weight'] = feature_weights.get(sug['feature'], 0.5)
                sug['efficiency'] = sug['confidence'] * sug['weight']
                
                all_suggestions.append(sug)
                
                if target_class == 0:
                    normal_tissue_suggestions.append(sug)
                elif target_class < original_prediction:
                    milder_suggestions.append(sug)
                else:
                    other_suggestions.append(sug)
        
        # 按效率排序
        all_suggestions.sort(key=lambda x: x.get('efficiency', 0), reverse=True)
        normal_tissue_suggestions.sort(key=lambda x: x.get('efficiency', 0), reverse=True)
        milder_suggestions.sort(key=lambda x: x.get('efficiency', 0), reverse=True)
        other_suggestions.sort(key=lambda x: x.get('efficiency', 0), reverse=True)
        
        # 如果没有找到任何建议，提供基于SHAP的通用建议
        if not all_suggestions:
            for i, feature in enumerate(feature_names):
                shap_val = shap_values[i]
                if abs(shap_val) > 0.001:
                    direction = "减少" if shap_val > 0 else "增加"
                    factor = 0.7 if shap_val > 0 else 1.3
                    
                    all_suggestions.append({
                        'feature': feature,
                        'change_factor': factor,
                        'confidence': 0.3,
                        'original_value': base_values[i],
                        'new_value': base_values[i] * factor,
                        'direction': direction,
                        'target_class': original_prediction,
                        'target_name': burn_type_mapping[original_prediction]['cn'],
                        'improvement': 0,
                        'weight': feature_weights.get(feature, 0.5),
                        'efficiency': 0.15,
                        'is_fallback': True
                    })
        
        return {
            'all_counterfactuals': all_suggestions[:10],
            'normal_tissue_suggestions': normal_tissue_suggestions[:3],
            'milder_suggestions': milder_suggestions[:3],
            'other_suggestions': other_suggestions[:3],
            'original_prediction': original_prediction,
            'feature_weights': feature_weights,
            'shap_directions': dict(zip(feature_names, shap_values)),
            'skip_analysis': False,
            'has_normal_tissue_suggestions': len(normal_tissue_suggestions) > 0
        }
        
    except Exception as e:
        st.warning(f"反事实分析遇到问题: {str(e)}")
        return {
            'all_counterfactuals': [],
            'normal_tissue_suggestions': [],
            'milder_suggestions': [],
            'other_suggestions': [],
            'original_prediction': original_prediction,
            'skip_analysis': False,
            'has_normal_tissue_suggestions': False
        }

# 优化的反事实分析可视化
def plot_optimized_counterfactual_analysis(counterfactual_results, burn_type_mapping):
    try:
        if not counterfactual_results or counterfactual_results.get('skip_analysis', False):
            return create_no_results_plot("无反事实分析结果")
        
        all_suggestions = counterfactual_results.get('all_counterfactuals', [])
        if not all_suggestions:
            return create_no_results_plot("未找到可行的特征调整方案")
        
        # 获取字体设置
        font_settings = get_chart_font_settings()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'反事实分析 - 烧伤等级改善策略 (当前: {burn_type_mapping[counterfactual_results["original_prediction"]]["cn"]})', 
                     fontsize=font_settings['title_font']['size'] + 2, fontweight='bold',
                     fontfamily=font_settings['title_font']['family'])
        
        # 提取各类建议
        normal_suggestions = counterfactual_results.get('normal_tissue_suggestions', [])
        milder_suggestions = counterfactual_results.get('milder_suggestions', [])
        other_suggestions = counterfactual_results.get('other_suggestions', [])
        
        # 图1: 正常组织恢复策略（如果有）
        ax1 = axes[0, 0]
        if normal_suggestions:
            plot_suggestion_chart(ax1, normal_suggestions[:3], "恢复正常组织策略", font_settings)
        else:
            ax1.text(0.5, 0.5, '⚠️ 未找到直接恢复为\n正常组织的策略', 
                     ha='center', va='center', transform=ax1.transAxes,
                     fontsize=font_settings['label_font']['size'],
                     fontfamily=font_settings['label_font']['family'])
            ax1.set_title('恢复正常组织策略', fontsize=font_settings['title_font']['size'], fontweight='bold',
                         fontfamily=font_settings['title_font']['family'])
        
        # 图2: 轻度烧伤改善策略
        ax2 = axes[0, 1]
        if milder_suggestions:
            plot_suggestion_chart(ax2, milder_suggestions[:3], "改善为更轻度烧伤策略", font_settings)
        else:
            ax2.text(0.5, 0.5, '⚠️ 未找到改善为\n更轻度烧伤的策略', 
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=font_settings['label_font']['size'],
                     fontfamily=font_settings['label_font']['family'])
            ax2.set_title('改善为更轻度烧伤策略', fontsize=font_settings['title_font']['size'], fontweight='bold',
                         fontfamily=font_settings['title_font']['family'])
        
        # 图3: 其他变化策略
        ax3 = axes[1, 0]
        if other_suggestions:
            plot_suggestion_chart(ax3, other_suggestions[:3], "其他变化策略", font_settings)
        else:
            ax3.text(0.5, 0.5, '⚠️ 未找到其他\n可行变化策略', 
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=font_settings['label_font']['size'],
                     fontfamily=font_settings['label_font']['family'])
            ax3.set_title('其他变化策略', fontsize=font_settings['title_font']['size'], fontweight='bold',
                         fontfamily=font_settings['title_font']['family'])
        
        # 图4: 改善程度总结
        ax4 = axes[1, 1]
        plot_improvement_summary(ax4, normal_suggestions, milder_suggestions, 
                               counterfactual_results['original_prediction'], burn_type_mapping, font_settings)
        
        apply_chart_font_settings(ax1)
        apply_chart_font_settings(ax2)
        apply_chart_font_settings(ax3)
        apply_chart_font_settings(ax4, xlabel='策略类型', ylabel='改善程度')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.warning(f"反事实图表绘制错误: {str(e)}")
        return create_error_plot(str(e), font_settings)

def plot_suggestion_chart(ax, suggestions, title, font_settings):
    """绘制建议图表"""
    if not suggestions:
        return
    
    features = [s['feature'] for s in suggestions]
    efficiencies = [s.get('efficiency', s.get('confidence', 0)) for s in suggestions]
    confidences = [s.get('confidence', 0) for s in suggestions]
    targets = [s.get('target_name', '未知') for s in suggestions]
    
    y_pos = np.arange(len(features))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    bars = ax.barh(y_pos, efficiencies, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    
    # 创建标签：特征 + 目标
    labels = []
    for feat, target in zip(features, targets):
        labels.append(f"{feat}\n→ {target}")
    
    ax.set_yticklabels(labels, fontfamily=font_settings['tick_font']['family'])
    ax.invert_yaxis()
    ax.set_xlabel('综合效率', fontweight='bold',
                  fontfamily=font_settings['axis_font']['family'],
                  fontsize=font_settings['axis_font']['size'])
    ax.set_title(title, fontsize=font_settings['title_font']['size'], fontweight='bold',
                 fontfamily=font_settings['title_font']['family'])
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, eff, conf, target) in enumerate(zip(bars, efficiencies, confidences, targets)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'效率: {eff:.3f}\n置信度: {conf:.1%}', 
                ha='left', va='center', fontweight='bold', 
                fontsize=font_settings['label_font']['size'] - 1,
                fontfamily=font_settings['label_font']['family'])

def plot_improvement_summary(ax, normal_suggestions, milder_suggestions, original_class, burn_type_mapping, font_settings):
    """绘制改善程度总结图"""
    categories = ['恢复正常组织', '改善为轻度烧伤', '其他变化']
    
    # 计算平均改善程度
    avg_normal_improvement = np.mean([s.get('improvement', 0) for s in normal_suggestions]) if normal_suggestions else 0
    avg_milder_improvement = np.mean([s.get('improvement', 0) for s in milder_suggestions]) if milder_suggestions else 0
    avg_other_improvement = 0  # 其他变化假设为0
    
    improvements = [avg_normal_improvement, avg_milder_improvement, avg_other_improvement]
    colors = ['#4CAF50', '#FF9800', '#9E9E9E']
    
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, improvements, color=colors, alpha=0.8, width=0.6)
    
    ax.set_xlabel('策略类型', fontweight='bold',
                  fontfamily=font_settings['axis_font']['family'],
                  fontsize=font_settings['axis_font']['size'])
    ax.set_ylabel('平均改善程度', fontweight='bold',
                  fontfamily=font_settings['axis_font']['family'],
                  fontsize=font_settings['axis_font']['size'])
    ax.set_title('策略改善程度总结', fontsize=font_settings['title_font']['size'], fontweight='bold',
                 fontfamily=font_settings['title_font']['family'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, fontfamily=font_settings['tick_font']['family'])
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{imp:.0f}分', ha='center', va='bottom',
                fontsize=font_settings['label_font']['size'], fontweight='bold',
                fontfamily=font_settings['label_font']['family'])
    
    # 添加当前状态
    ax.text(0.5, -0.2, f'当前状态: {burn_type_mapping[original_class]["cn"]}',
            transform=ax.transAxes, ha='center', 
            fontsize=font_settings['label_font']['size'],
            fontfamily=font_settings['label_font']['family'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))

def create_no_results_plot(message):
    """创建无结果消息图"""
    font_settings = get_chart_font_settings()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0.5, 0.6, '⚠️ 无反事实分析结果', 
            ha='center', va='center', fontsize=font_settings['title_font']['size'],
            fontweight='bold', color='#FF9800',
            fontfamily=font_settings['title_font']['family'])
    ax.text(0.5, 0.4, message,
            ha='center', va='center', fontsize=font_settings['label_font']['size'],
            fontfamily=font_settings['label_font']['family'])
    return fig

def create_error_plot(error_msg, font_settings):
    """创建错误消息图"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.text(0.5, 0.6, '❌ 图表绘制错误', 
            ha='center', va='center', fontsize=font_settings['title_font']['size'],
            fontweight='bold', color='#F44336',
            fontfamily=font_settings['title_font']['family'])
    ax.text(0.5, 0.4, error_msg[:50] + '...' if len(error_msg) > 50 else error_msg,
            ha='center', va='center', fontsize=font_settings['label_font']['size'],
            fontfamily=font_settings['label_font']['family'])
    return fig


# 生成医疗检测报告的函数 - 修改1：增强报告内容
def generate_medical_report(input_data, prediction, probabilities, shap_results, graph_results, counterfactual_results, burn_type_mapping, feature_names, language='中文'):
    """生成详细的医疗检测报告"""
    
    burn_info = burn_type_mapping[prediction]
    
    if language == '中文':
        report = f"""烧伤智能识别系统 - 医疗检测报告
==================================================

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

【基本信息】
患者样本编号: {pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}
分析模型: 随机森林多分类模型
数据精度: 小数点后10位

【输入参数详细数据】
BG1 (生物标志物1): {input_data.iloc[0, 0]:.10f}
IL-1β (白细胞介素-1β): {input_data.iloc[0, 1]:.10f} pg/mL
EGF (表皮生长因子): {input_data.iloc[0, 2]:.10f} pg/mL
BG2 (生物标志物2): {input_data.iloc[0, 3]:.10f}

【诊断结果】
主要诊断: {burn_info['cn']} ({burn_info['en']})
置信度: {probabilities[prediction]:.2%}
临床描述: {burn_info['description']}

【概率分布分析】
"""
        for i, prob in enumerate(probabilities):
            report += f"{burn_type_mapping[i]['cn']}: {prob:.2%}\n"
        
        report += f"\n【生物标志物临床意义分析】\n"
        report += "="*50 + "\n"
        
        # 基于SHAP值的临床分析
        if shap_results:
            shap_values = shap_results['shap_values']
            for i, feature in enumerate(feature_names):
                shap_val = shap_values[i]
                original_val = input_data.iloc[0, i]
                
                report += f"\n{feature}分析:\n"
                report += f"- 当前水平: {original_val:.10f}\n"
                report += f"- 对诊断影响: {shap_val:+.6f} "
                
                if shap_val > 0.01:
                    report += "(显著正向影响 → 促进该诊断)\n"
                elif shap_val < -0.01:
                    report += "(显著负向影响 → 抑制该诊断)\n"
                else:
                    report += "(影响较小)\n"
                
                # 针对每个特征的临床解释
                if feature == "IL-1β":
                    if original_val > 500:
                        report += "- 临床意义: IL-1β水平显著升高，表明强烈的炎症反应，可能与严重烧伤相关\n"
                        report += "- 治疗建议: 需要积极抗炎治疗，监测全身炎症反应综合征\n"
                        report += "- 日常注意: 避免感染，保持伤口清洁，定期监测炎症指标\n"
                    elif original_val > 300:
                        report += "- 临床意义: IL-1β水平中度升高，提示中度炎症状态\n"
                        report += "- 治疗建议: 适当抗炎治疗，密切观察病情变化\n"
                        report += "- 日常注意: 注意伤口护理，避免刺激性物质接触\n"
                    else:
                        report += "- 临床意义: IL-1β水平在正常范围内\n"
                        report += "- 治疗建议: 维持当前治疗方案\n"
                        report += "- 日常注意: 继续保持良好的伤口护理习惯\n"
                        
                elif feature == "EGF":
                    if original_val < 400:
                        report += "- 临床意义: EGF水平偏低，可能影响伤口愈合能力\n"
                        report += "- 治疗建议: 考虑外源性EGF补充治疗\n"
                        report += "- 日常注意: 加强营养支持，促进内源性EGF生成\n"
                    elif original_val > 600:
                        report += "- 临床意义: EGF水平较高，有利于组织修复\n"
                        report += "- 治疗建议: 维持良好的愈合环境\n"
                        report += "- 日常注意: 继续保持有利于伤口愈合的生活方式\n"
                    else:
                        report += "- 临床意义: EGF水平在正常范围内\n"
                        report += "- 治疗建议: 当前EGF水平适宜伤口愈合\n"
                        report += "- 日常注意: 保持均衡营养，促进正常愈合\n"
                        
                elif feature == "BG1":
                    if abs(original_val) > 3:
                        report += "- 临床意义: BG1水平异常，可能指示组织损伤\n"
                        report += "- 治疗建议: 进一步评估组织损伤程度\n"
                        report += "- 日常注意: 避免进一步组织损伤，注意保护创面\n"
                    else:
                        report += "- 临床意义: BG1水平在参考范围内\n"
                        report += "- 治疗建议: 继续当前治疗\n"
                        report += "- 日常注意: 定期监测生物标志物变化\n"
                        
                elif feature == "BG2":
                    if original_val < -0.5:
                        report += "- 临床意义: BG2水平显著偏低，提示修复能力受损\n"
                        report += "- 治疗建议: 加强修复支持治疗\n"
                        report += "- 日常注意: 注意营养补充，促进修复能力恢复\n"
                    elif original_val > 0.5:
                        report += "- 临床意义: BG2水平偏高，可能反映代偿性修复\n"
                        report += "- 治疗建议: 观察修复进展，适时调整治疗\n"
                        report += "- 日常注意: 维持适度的修复环境\n"
                    else:
                        report += "- 临床意义: BG2水平在正常波动范围内\n"
                        report += "- 治疗建议: 当前修复状态良好\n"
                        report += "- 日常注意: 继续保持有利于修复的生活方式\n"
        
        # SHAP分析结果
        if shap_results:
            report += f"\n【SHAP可解释性分析】\n"
            report += "="*50 + "\n"
            report += "特征重要性排序 (基于SHAP绝对值):\n"
            for i, (feature, importance) in enumerate(zip(shap_results['sorted_features'], shap_results['sorted_importance'])):
                report += f"{i+1}. {feature}: {importance:.10f}\n"
            # 新增：SHAP图表分析描述
            report += f"\n【SHAP图表分析解读】\n"
            report += "多类别SHAP分析图显示所有六种烧伤类型的特征影响模式：\n"
            report += "- 红色条形表示特征对当前诊断有正向促进作用\n"
            report += "- 蓝色条形表示特征对当前诊断有负向抑制作用\n"
            report += "- 散点大小反映特征重要性绝对值大小\n"
            report += f"- 当前诊断类别({burn_info['cn']})用红色边框高亮显示\n"
            report += f"- 最重要的特征: {shap_results['sorted_features'][0]} (SHAP值: {shap_results['sorted_importance'][0]:.6f})\n"
        
        # 图网络分析结果
        if graph_results:
            report += f"\n【图网络分析结果】\n"
            report += "="*50 + "\n"
            report += f"网络节点数: {len(graph_results['graph'].nodes())}\n"
            report += f"网络边数: {len(graph_results['graph'].edges())}\n"
            report += "特征中心性分析:\n"
            for feature in graph_results['degree_centrality']:
                report += f"- {feature}: 度中心性={graph_results['degree_centrality'][feature]:.6f}, 介数中心性={graph_results['betweenness_centrality'][feature]:.6f}, 紧密中心性={graph_results['closeness_centrality'][feature]:.6f}\n"
            
            # 新增：图网络图表分析描述
            report += f"\n【图网络图表分析解读】\n"
            report += "特征关联网络图揭示生物标志物间的相互作用关系：\n"
            report += "- 节点大小反映特征在预测中的相对重要性\n"
            report += "- 边粗细表示特征间相关性强度\n"
            report += "- 度中心性高的特征在网络中连接更广泛\n"
            report += "- 介数中心性高的特征在网络中起桥梁作用\n"
            report += "- 雷达图展示不同中心性指标的对比分析\n"
            report += "- 热力图直观显示特征间的数值相关性\n"
            
            # 分析网络结构特点
            max_degree_feature = max(graph_results['degree_centrality'], key=graph_results['degree_centrality'].get)
            max_betweenness_feature = max(graph_results['betweenness_centrality'], key=graph_results['betweenness_centrality'].get)
            report += f"- 网络枢纽特征: {max_degree_feature} (连接最广泛)\n"
            report += f"- 关键桥梁特征: {max_betweenness_feature} (信息传递关键节点)\n"
        
        # 反事实分析结果 - 修改后的判断条件
        if (counterfactual_results and 
            not counterfactual_results.get('skip_analysis', False) and
            (counterfactual_results.get('has_normal_tissue_suggestions', False) or
             counterfactual_results.get('normal_tissue_suggestions', []))):
            
            suggestions = counterfactual_results.get('normal_tissue_suggestions', [])
            if suggestions:
                report += f"\n【反事实分析与治疗建议】\n"
                report += "="*50 + "\n"
                report += "基于模型预测的干预策略分析:\n\n"
                
                for i, suggestion in enumerate(suggestions[:3], 1):
                    report += f"治疗方案 {i}:\n"
                    report += f"- 调整目标: 将{suggestion.get('feature', '未知特征')}{suggestion.get('direction', '调整')}到原来的 {suggestion.get('change_factor', 1.0):.1f}倍\n"
                    report += f"- 具体数值: {suggestion.get('original_value', 0):.10f} → {suggestion.get('new_value', 0):.10f}\n"
                    report += f"- 预期效果置信度: {suggestion.get('confidence', 0):.2%}\n"
                    report += f"- 临床意义: 预测从{burn_type_mapping[counterfactual_results.get('original_prediction', 0)]['cn']}改善到{burn_type_mapping[suggestion.get('target_class', 0)]['cn']}\n\n"
                
                # 新增：反事实图表分析描述
                report += f"\n【反事实图表分析解读】\n"
                report += "反事实分析图展示特征调整对诊断结果的影响：\n"
                report += "- 左侧条形图显示不同调整方案的预期置信度\n"
                report += "- 绿色表示增加特征值，红色表示减少特征值\n"
                report += "- 右侧路径图对比当前值与目标值的差异\n"
                report += "- 箭头方向指示特征调整的方向和幅度\n"
                if suggestions:
                    report += f"- 最优调整方案: {suggestions[0].get('feature', '未知特征')} ({suggestions[0].get('direction', '调整')}{suggestions[0].get('change_factor', 1.0):.1f}倍)\n"
                    
        # 治疗和注意事项
        report += f"\n【临床治疗建议与注意事项】\n"
        report += "="*50 + "\n"
        
        if prediction == 0:
            report += "当前诊断为正常组织，无需特殊治疗。\n"
            report += "建议:\n"
            report += "- 定期监测生物标志物水平\n"
            report += "- 保持健康生活方式\n"
            report += "- 避免烧伤风险因素\n"
        else:
            report += f"针对{burn_info['cn']}的治疗建议:\n"
            
            if prediction in [1, 2]:  # 浅表和深部部分厚度烧伤
                report += "- 立即进行伤口清洁和消毒\n"
                report += "- 使用适当的敷料保护创面\n"
                report += "- 考虑使用生长因子促进愈合\n"
                report += "- 定期更换敷料，监测感染迹象\n"
                report += "- 如IL-1β水平高，考虑抗炎治疗\n"
                
            elif prediction == 3:  # 全层厚度烧伤
                report += "- 需要外科清创和植皮手术\n"
                report += "- 全身抗感染治疗\n"
                report += "- 营养支持，促进组织修复\n"
                report += "- 疼痛管理和炎症控制\n"
                report += "- 长期康复和功能训练\n"
                
            elif prediction == 4:  # 电击烧伤
                report += "- 评估深部组织损伤程度\n"
                report += "- 监测心电图和肌酸激酶\n"
                report += "- 积极清创，预防感染\n"
                report += "- 注意可能的并发症\n"
                report += "- 多学科团队协作治疗\n"
                
            elif prediction == 5:  # 火焰烧伤
                report += "- 评估吸入性损伤风险\n"
                report += "- 全面清创和烧伤护理\n"
                report += "- 预防感染和败血症\n"
                report += "- 营养支持和代谢管理\n"
                report += "- 心理支持和康复治疗\n"
            
            report += "\n日常注意事项:\n"
            report += "- 严格遵医嘱进行治疗\n"
            report += "- 定期复查生物标志物\n"
            report += "- 注意伤口护理和个人卫生\n"
            report += "- 合理营养，促进愈合\n"
            report += "- 避免刺激性物质接触创面\n"
        
        report += f"\n【报告说明】\n"
        report += "="*50 + "\n"
        report += "1. 本报告基于机器学习模型分析生成，仅供参考\n"
        report += "2. 临床诊断需结合临床表现和医师判断\n"
        report += "3. 治疗建议需在专业医师指导下实施\n"
        report += "4. 定期随访和监测对治疗效果至关重要\n"
        
    else:  # English version
        report = f"""Burn Intelligent Recognition System - Medical Analysis Report
==================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

【Basic Information】
Sample ID: {pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}
Analysis Model: Random Forest Multi-class Model
Data Precision: 10 decimal places

【Input Parameters】
BG1 (Biomarker 1): {input_data.iloc[0, 0]:.10f}
IL-1β (Interleukin-1β): {input_data.iloc[0, 1]:.10f} pg/mL
EGF (Epidermal Growth Factor): {input_data.iloc[0, 2]:.10f} pg/mL
BG2 (Biomarker 2): {input_data.iloc[0, 3]:.10f}

【Diagnosis Results】
Primary Diagnosis: {burn_info['en']} ({burn_info['cn']})
Confidence: {probabilities[prediction]:.2%}
Clinical Description: {burn_info['description_en']}

【Probability Distribution Analysis】
"""
        for i, prob in enumerate(probabilities):
            report += f"{burn_type_mapping[i]['en']}: {prob:.2%}\n"
    
    return report

# 自动加载模型
if 'model' not in st.session_state:
    with st.spinner("正在加载模型..."): st.session_state.model = load_model()

# 烧伤类型映射
burn_type_mapping = {
    0: {"en": "Normal", "cn": "正常组织", "color": "#4CAF50", "description": "正常皮肤组织", "description_en": "Normal skin tissue"},
    1: {"en": "Superficial partial-thickness", "cn": "浅表部分厚度烧伤", "color": "#FF9800", "description": "表皮和部分真皮受损", "description_en": "Epidermis and partial dermis damage"},
    2: {"en": "Deep partial-thickness", "cn": "深层部分厚度烧伤", "color": "#FF5722", "description": "真皮深层受损", "description_en": "Deep dermis damage"},
    3: {"en": "Full-thickness", "cn": "全层厚度烧伤", "color": "#F44336", "description": "皮肤全层受损", "description_en": "Full-thickness skin damage"},
    4: {"en": "Electrical", "cn": "电击烧伤", "color": "#9C27B0", "description": "电击导致的组织损伤", "description_en": "Tissue damage caused by electric shock"},
    5: {"en": "Flame", "cn": "火焰烧伤", "color": "#795548", "description": "火焰直接接触导致的烧伤", "description_en": "Burn caused by direct flame contact"}
}

# 初始化session state
if 'language' not in st.session_state: st.session_state.language = '中文'
if 'chart_colors' not in st.session_state: st.session_state.chart_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
if 'title_font' not in st.session_state: st.session_state.title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
if 'label_font' not in st.session_state: st.session_state.label_font = {'family': 'Microsoft YaHei', 'size': 10}
if 'theme' not in st.session_state: st.session_state.theme = 'light'
if 'data_precision' not in st.session_state: st.session_state.data_precision = 10

# 初始化图表字体设置
if 'chart_title_font' not in st.session_state: 
    st.session_state.chart_title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
if 'chart_axis_font' not in st.session_state: 
    st.session_state.chart_axis_font = {'family': 'Microsoft YaHei', 'size': 10}
if 'chart_tick_font' not in st.session_state: 
    st.session_state.chart_tick_font = {'family': 'Microsoft YaHei', 'size': 8}
if 'chart_label_font' not in st.session_state: 
    st.session_state.chart_label_font = {'family': 'Microsoft YaHei', 'size': 9}

# 侧边栏
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/fire-element.png", width=80)
    st.title("烧伤识别系统")
    st.markdown("---")
    # 修复：添加唯一的key参数
    app_mode = st.selectbox("选择应用模式", ["🔬 烧伤识别分析", "📖 使用指南", "⚙️ 系统设置"], key="app_mode_select")
    st.markdown("---")
    if st.session_state.model is not None: st.success("✅ 模型已加载")
    else: st.error("❌ 模型加载失败")

# 主页面内容
if app_mode == "🔬 烧伤识别分析":
    st.markdown('<div class="main-header">🔥 烧伤智能识别与分析系统</div>', unsafe_allow_html=True)
    
    if st.session_state.model is not None:
        model = st.session_state.model
        st.success("✅ 专业模式 - 使用训练好的随机森林模型")
    else:
        st.error("❌ 模型加载失败，无法进行分析")
        st.stop()
    
    tab1, tab2 = st.tabs(["🔍 单样本分析", "📊 批量分析"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.language == '中文':
                st.markdown('<div class="sub-header">📋 输入烧伤特征参数</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sub-header">📋 Input Burn Characteristics</div>', unsafe_allow_html=True)
            
            with st.form("input_form"):
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    feature1 = st.number_input("BG1 生物标志物", value=-3.696319906, format="%.10f", help="第一个生物标志物参数")
                    feature2 = st.number_input("IL-1β (pg/mL)", value=387.7812826, format="%.10f", help="白细胞介素-1β浓度")
                with col1_2:
                    feature3 = st.number_input("EGF (pg/mL)", value=1060.934711, format="%.10f", help="表皮生长因子浓度")
                    feature4 = st.number_input("BG2 生物标志物", value=-0.501551816, format="%.10f", help="第二个生物标志物参数")
                
                if st.session_state.language == '中文':
                    advanced_analysis = st.checkbox("执行SHAP+图网络+反事实分析", value=True, key="advanced_checkbox")
                    submitted = st.form_submit_button("🚀 开始分析", use_container_width=True)
                else:
                    advanced_analysis = st.checkbox("Perform SHAP+Graph+Counterfactual Analysis", value=True, key="advanced_checkbox_en")
                    submitted = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
        
        with col2:
            if st.session_state.language == '中文':
                st.markdown('<div class="sub-header">💡 参数说明</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="feature-box"><strong>BG1:</strong> 关键生物标志物1，反映组织炎症状态</div>
                <div class="feature-box"><strong>IL-1β:</strong> 炎症因子，浓度与烧伤严重程度相关</div>
                <div class="feature-box"><strong>EGF:</strong> 表皮生长因子，促进伤口愈合</div>
                <div class="feature-box"><strong>BG2:</strong> 关键生物标志物2，组织修复指标</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="sub-header">💡 Parameter Description</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="feature-box"><strong>BG1:</strong> Key biomarker 1, reflects tissue inflammation status</div>
                <div class="feature-box"><strong>IL-1β:</strong> Inflammatory factor, concentration correlates with burn severity</div>
                <div class="feature-box"><strong>EGF:</strong> Epidermal growth factor, promotes wound healing</div>
                <div class="feature-box"><strong>BG2:</strong> Key biomarker 2, tissue repair indicator</div>
                """, unsafe_allow_html=True)
        
        if submitted:
            try:
                input_data = pd.DataFrame([[feature1, feature2, feature3, feature4]], columns=model.feature_names_in_)
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                if st.session_state.language == '中文':
                    st.markdown('<div class="sub-header">📊 分析结果</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                with col_res2:
                    burn_info = burn_type_mapping[prediction]
                    if st.session_state.language == '中文':
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>诊断结果: {burn_info['cn']}</h3>
                            <p><strong>英文名称:</strong> {burn_info['en']}</p>
                            <p><strong>描述:</strong> {burn_info['description']}</p>
                            <p><strong>置信度:</strong> {probabilities[prediction]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Diagnosis Result: {burn_info['en']}</h3>
                            <p><strong>Chinese Name:</strong> {burn_info['cn']}</p>
                            <p><strong>Description:</strong> {burn_info['description_en']}</p>
                            <p><strong>Confidence:</strong> {probabilities[prediction]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if advanced_analysis:
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">🔬 高级模型分析</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">🔬 Advanced Model Analysis</div>', unsafe_allow_html=True)
                    
                    # SHAP分析
                    with st.spinner("正在进行SHAP分析..." if st.session_state.language == '中文' else "Performing SHAP analysis..."):
                        shap_results = perform_shap_analysis(model, input_data, model.feature_names_in_)
                    
                    # 图网络分析
                    with st.spinner("正在进行图网络分析..." if st.session_state.language == '中文' else "Performing graph network analysis..."):
                        graph_results = perform_graph_analysis([feature1, feature2, feature3, feature4], model.feature_names_in_, prediction, burn_type_mapping)
                    
                    # 在调用反事实分析之前添加调试
                    st.write("🔍 调试信息：")
                    st.write(f"SHAP结果是否为空: {shap_results is not None}")
                    st.write(f"图网络结果是否为空: {graph_results is not None}")
                    if shap_results:
                        st.write(f"SHAP值: {shap_results.get('shap_values', [])}")
                    if graph_results:
                        st.write(f"图网络有{len(graph_results.get('graph', nx.Graph()).nodes())}个节点")

                    # 反事实分析 - 传入SHAP和图网络结果
                    if prediction != 0:  # 不是正常组织
                        with st.spinner("正在进行反事实分析..." if st.session_state.language == '中文' else "Performing counterfactual analysis..."):
                            counterfactual_results = perform_counterfactual_analysis(
                                model=model,
                                input_data=input_data,
                                original_prediction=prediction,
                                feature_names=model.feature_names_in_,
                                burn_type_mapping=burn_type_mapping,
                                shap_results=shap_results,  # 关键：传入SHAP结果
                                graph_results=graph_results  # 关键：传入图网络结果
                            )
                            
                            # 添加调试信息
                            st.write("🔍 反事实分析结果:")
                            st.write(f"是否有正常组织建议: {counterfactual_results.get('has_normal_tissue_suggestions', False)}")
                            st.write(f"正常组织建议数量: {len(counterfactual_results.get('normal_tissue_suggestions', []))}")
                            st.write(f"轻度烧伤建议数量: {len(counterfactual_results.get('milder_suggestions', []))}")
                            st.write(f"所有建议数量: {len(counterfactual_results.get('all_counterfactuals', []))}")
                    else:
                        # 正常组织也需要调用反事实分析，但会返回维持建议
                        with st.spinner("生成维持建议..." if st.session_state.language == '中文' else "Generating maintenance suggestions..."):
                            counterfactual_results = perform_counterfactual_analysis(
                                model=model,
                                input_data=input_data,
                                original_prediction=prediction,
                                feature_names=model.feature_names_in_,
                                burn_type_mapping=burn_type_mapping,
                                shap_results=shap_results,
                                graph_results=graph_results
                            )
                            
                            # 添加调试信息
                            st.write("🔍 反事实分析结果（正常组织）:")
                            st.write(f"是否有正常组织建议: {counterfactual_results.get('has_normal_tissue_suggestions', False)}")
                            st.write(f"正常组织建议数量: {len(counterfactual_results.get('normal_tissue_suggestions', []))}")
                    
                    # 显示SHAP分析结果
                    if shap_results:
                        if st.session_state.language == '中文':
                            st.markdown("##### 📈 SHAP多类别分析")
                        else:
                            st.markdown("##### 📈 SHAP Multi-Class Analysis")
                        
                        col_shap1, col_shap2 = st.columns([1, 1])
                        
                        with col_shap1:
                            # 图1: 合并的SHAP分析图表
                            fig_combined = plot_combined_shap_analysis(shap_results, model.feature_names_in_, burn_type_mapping)
                            if fig_combined:
                                st.pyplot(fig_combined)
                                if st.session_state.language == '中文':
                                    st.caption("图1: SHAP合并分析 - 特征影响方向和重要性")
                                else:
                                    st.caption("Figure 1: Combined SHAP Analysis - Feature Impact and Importance")
                        
                        with col_shap2:
                            # 图2: 当前预测类别的特征重要性图
                            fig_current = plot_current_prediction_shap(shap_results, model.feature_names_in_, burn_type_mapping)
                            if fig_current:
                                st.pyplot(fig_current)
                                if st.session_state.language == '中文':
                                    st.caption("图2: 当前预测类别特征重要性分析")
                                else:
                                    st.caption("Figure 2: Feature Importance for Current Prediction")
                    
                    # 显示图网络分析结果
                    if graph_results:
                        if st.session_state.language == '中文':
                            st.markdown("##### 🔗 特征关联图网络分析")
                        else:
                            st.markdown("##### 🔗 Feature Correlation Graph Analysis")
                        
                        graph_fig = plot_optimized_graph_analysis(graph_results, model.feature_names_in_, burn_info)
                        if graph_fig:
                            st.pyplot(graph_fig)
                    
                    # 显示反事实分析结果 - 修改判断条件
                    if (counterfactual_results and 
                        not counterfactual_results.get('skip_analysis', False)):
                        
                        # 检查是否有任何建议
                        has_suggestions = (
                            counterfactual_results.get('has_normal_tissue_suggestions', False) or
                            counterfactual_results.get('normal_tissue_suggestions', []) or
                            counterfactual_results.get('milder_suggestions', []) or
                            counterfactual_results.get('all_counterfactuals', [])
                        )
                        
                        if has_suggestions:
                            if st.session_state.language == '中文':
                                st.markdown("##### 🔄 烧伤等级改善策略分析")
                            else:
                                st.markdown("##### 🔄 Burn Level Improvement Strategy Analysis")
                            
                            # 显示图表
                            counterfactual_fig = plot_optimized_counterfactual_analysis(counterfactual_results, burn_type_mapping)
                            if counterfactual_fig:
                                st.pyplot(counterfactual_fig)
                            
                            # 显示文字建议
                            if st.session_state.language == '中文':
                                if counterfactual_results.get('has_normal_tissue_suggestions', False):
                                    suggestions = counterfactual_results.get('normal_tissue_suggestions', [])
                                    if suggestions:
                                        st.markdown("###### 💡 恢复到正常组织的调整建议:")
                                        for i, suggestion in enumerate(suggestions[:3], 1):
                                            st.markdown(f"""
                                            <div class="analysis-box">
                                            <strong>方案 {i}:</strong> 将 <strong>{suggestion.get('feature', '未知')}</strong> {suggestion.get('direction', '调整')}到原来的 <strong>{suggestion.get('change_factor', 1.0):.1f}倍</strong><br>
                                            - 原始值: {suggestion.get('original_value', 0):.10f} → 目标值: {suggestion.get('new_value', 0):.10f}<br>
                                            - 预测置信度: {suggestion.get('confidence', 0):.2%}<br>
                                            - 改善程度: {suggestion.get('improvement', 0):.0f}分<br>
                                            - 效果: 预测结果从 <strong>{burn_type_mapping[counterfactual_results.get('original_prediction', 0)]['cn']}</strong> 恢复到 <strong>正常组织</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                elif counterfactual_results.get('milder_suggestions', []):
                                    suggestions = counterfactual_results.get('milder_suggestions', [])
                                    if suggestions:
                                        st.markdown("###### 💡 改善为更轻度烧伤的建议:")
                                        for i, suggestion in enumerate(suggestions[:3], 1):
                                            st.markdown(f"""
                                            <div class="analysis-box">
                                            <strong>方案 {i}:</strong> 将 <strong>{suggestion.get('feature', '未知')}</strong> {suggestion.get('direction', '调整')}到原来的 <strong>{suggestion.get('change_factor', 1.0):.1f}倍</strong><br>
                                            - 原始值: {suggestion.get('original_value', 0):.10f} → 目标值: {suggestion.get('new_value', 0):.10f}<br>
                                            - 预测置信度: {suggestion.get('confidence', 0):.2%}<br>
                                            - 改善程度: 从 <strong>{burn_type_mapping[counterfactual_results.get('original_prediction', 0)]['cn']}</strong> 改善到 <strong>{suggestion.get('target_name', '未知')}</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                            else:
                                # 英文版本
                                if counterfactual_results.get('has_normal_tissue_suggestions', False):
                                    suggestions = counterfactual_results.get('normal_tissue_suggestions', [])
                                    if suggestions:
                                        st.markdown("###### 💡 Adjustment suggestions to restore normal tissue:")
                                        for i, suggestion in enumerate(suggestions[:3], 1):
                                            st.markdown(f"""
                                            <div class="analysis-box">
                                            <strong>Scenario {i}:</strong> Change <strong>{suggestion.get('feature', 'Unknown')}</strong> to <strong>{suggestion.get('change_factor', 1.0):.1f}x</strong> of original<br>
                                            - Original value: {suggestion.get('original_value', 0):.10f} → Target value: {suggestion.get('new_value', 0):.10f}<br>
                                            - Prediction confidence: {suggestion.get('confidence', 0):.2%}<br>
                                            - Improvement score: {suggestion.get('improvement', 0):.0f} points<br>
                                            - Effect: Prediction changes from <strong>{burn_type_mapping[counterfactual_results.get('original_prediction', 0)]['en']}</strong> to <strong>Normal Tissue</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                elif counterfactual_results.get('milder_suggestions', []):
                                    suggestions = counterfactual_results.get('milder_suggestions', [])
                                    if suggestions:
                                        st.markdown("###### 💡 Suggestions to improve to milder burn:")
                                        for i, suggestion in enumerate(suggestions[:3], 1):
                                            st.markdown(f"""
                                            <div class="analysis-box">
                                            <strong>Scenario {i}:</strong> Change <strong>{suggestion.get('feature', 'Unknown')}</strong> to <strong>{suggestion.get('change_factor', 1.0):.1f}x</strong> of original<br>
                                            - Original value: {suggestion.get('original_value', 0):.10f} → Target value: {suggestion.get('new_value', 0):.10f}<br>
                                            - Prediction confidence: {suggestion.get('confidence', 0):.2%}<br>
                                            - Improvement: From <strong>{burn_type_mapping[counterfactual_results.get('original_prediction', 0)]['en']}</strong> to <strong>{suggestion.get('target_name', 'Unknown')}</strong>
                                            </div>
                                            """, unsafe_allow_html=True)
                        
                        else:
                            if st.session_state.language == '中文':
                                st.info("⚠️ 未找到可行的特征调整方案，建议结合临床评估进行个体化治疗。")
                            else:
                                st.info("⚠️ No feasible feature adjustment solutions found. Consider personalized treatment based on clinical evaluation.")
                    
                    # 概率分布图
                    st.markdown("---")
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">📈 概率分布分析</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">📈 Probability Distribution Analysis</div>', unsafe_allow_html=True)
                    
                    # 获取字体设置
                    font_settings = get_chart_font_settings()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    if st.session_state.language == '中文':
                        title1, title2, ylabel = '烧伤类型概率分布', '概率分布饼图', '概率'
                        labels = [burn_type_mapping[i]['cn'] for i in range(len(probabilities))]
                    else:
                        title1, title2, ylabel = 'Burn Type Probability Distribution', 'Probability Distribution Pie Chart', 'Probability'
                        labels = [burn_type_mapping[i]['en'] for i in range(len(probabilities))]
                    
                    colors = st.session_state.chart_colors[:len(probabilities)]
                    bars = ax1.bar(range(len(probabilities)), probabilities, color=colors)
                    ax1.set_title(title1, fontfamily=font_settings['title_font']['family'],
                                 fontsize=font_settings['title_font']['size'])
                    ax1.set_xticks(range(len(probabilities)))
                    ax1.set_xticklabels(labels, rotation=45, ha='right', 
                                       fontfamily=font_settings['tick_font']['family'])
                    ax1.set_ylabel(ylabel, fontfamily=font_settings['axis_font']['family'],
                                 fontsize=font_settings['axis_font']['size'])
                    ax1.set_ylim(0, 1)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.1%}', 
                                ha='center', va='bottom', fontfamily=font_settings['label_font']['family'])
                    
                    ax2.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
                           textprops={'fontfamily': font_settings['label_font']['family']})
                    ax2.set_title(title2, fontfamily=font_settings['title_font']['family'],
                                 fontsize=font_settings['title_font']['size'])
                    
                    # 应用字体设置
                    apply_chart_font_settings(ax1, title=title1, ylabel=ylabel)
                    apply_chart_font_settings(ax2, title=title2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 结果导出 - 修改1：使用增强的医疗报告
                    st.markdown("---")
                    if st.session_state.language == '中文':
                        st.markdown('<div class="sub-header">💾 结果导出</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
                    
                    # 生成增强的医疗报告
                    report_text = generate_medical_report(input_data, prediction, probabilities, shap_results, graph_results, counterfactual_results, burn_type_mapping, model.feature_names_in_, st.session_state.language)
                    
                    col_exp1, col_exp2, col_exp3 = st.columns(3)
                    with col_exp1:
                        csv_data = input_data.copy()
                        csv_data['预测类型' if st.session_state.language == '中文' else 'Predicted Type'] = burn_info['cn' if st.session_state.language == '中文' else 'en']
                        csv_data['置信度' if st.session_state.language == '中文' else 'Confidence'] = probabilities[prediction]
                        csv = csv_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 导出CSV" if st.session_state.language == '中文' else "📥 Export CSV",
                            data=csv, file_name="burn_analysis_result.csv", mime="text/csv", use_container_width=True
                        )
                    with col_exp2:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(
                            label="🖼️ 导出图表" if st.session_state.language == '中文' else "🖼️ Export Chart",
                            data=buf.getvalue(), file_name="burn_analysis_chart.png", mime="image/png", use_container_width=True
                        )
                    with col_exp3:
                        st.download_button(
                            label="📄 导出医疗报告" if st.session_state.language == '中文' else "📄 Export Medical Report",
                            data=report_text.encode('utf-8'), file_name="burn_medical_report.txt", mime="text/plain", use_container_width=True
                        )
                    
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")

    with tab2:
        if st.session_state.language == '中文':
            st.markdown('<div class="sub-header">📁 批量数据处理</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sub-header">📁 Batch Data Processing</div>', unsafe_allow_html=True)
        st.info("批量分析功能")

elif app_mode == "📖 使用指南":
    st.markdown('<div class="main-header">📖 使用指南</div>', unsafe_allow_html=True)
    
    # 创建标签页
    tab_guide1, tab_guide2, tab_guide3, tab_guide4, tab_guide5 = st.tabs(["📋 系统介绍", "🔬 使用步骤", "📊 数据说明", "🧠 算法原理", "❓ 常见问题"])
    
    with tab_guide1:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## 🔬 系统介绍")
        st.markdown("""
        本系统基于机器学习算法，通过对生物标志物的分析，实现烧伤类型的智能识别和分类。系统集成了先进的模型可解释性技术，
        包括SHAP分析、图网络分析和反事实分析，为医疗专业人员提供全面的决策支持。
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_intro1, col_intro2 = st.columns(2)
        with col_intro1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 系统特色")
            st.markdown("""
            - **智能识别**: 基于随机森林算法的多分类模型
            - **可解释性**: 集成SHAP、图网络、反事实分析
            - **高精度**: 支持小数点后10位的数据精度
            - **可视化**: 丰富的图表和交互界面
            - **多语言**: 支持中英文界面切换
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_intro2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown("### 📊 功能模块")
            st.markdown("""
            - **单样本分析**: 单个样本的详细分析
            - **批量分析**: 批量数据处理功能
            - **高级分析**: SHAP+图网络+反事实分析
            - **结果导出**: 支持CSV、图表、报告导出
            - **系统设置**: 个性化界面配置
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab_guide2:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## 🔬 使用步骤")
        
        st.markdown("### 1. 单样本分析")
        steps = [
            ("📋 输入参数", "在单样本分析页面输入四个生物标志物的数值"),
            ("🚀 开始分析", "点击开始分析按钮获取预测结果"),
            ("📊 查看结果", "查看诊断结果、概率分布和置信度"),
            ("🔬 高级分析", "可选执行SHAP、图网络、反事实分析"),
            ("💾 导出结果", "导出CSV、图表和分析报告")
        ]
        
        for i, (step, desc) in enumerate(steps, 1):
            with st.expander(f"步骤 {i}: {step}"):
                st.markdown(desc)
        
        st.markdown("### 2. 批量分析")
        st.markdown("""
        - 准备包含BG1、EGF、IL-1β、BG2列的CSV或Excel文件
        - 在批量分析页面上传文件
        - 系统自动处理所有数据并生成分析报告
        - 下载批量分析结果
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab_guide3:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## 📊 数据说明")
        
        st.markdown("### 🔬 生物标志物参数")
        biomarkers = [
            ("BG1", "生物标志物1", "反映组织炎症状态的关键指标", "2.453646"),
            ("IL-1β", "白细胞介素-1β", "炎症因子，浓度与烧伤严重程度正相关", "340.098941 pg/mL"),
            ("EGF", "表皮生长因子", "促进伤口愈合的重要因子", "535.07482 pg/mL"),
            ("BG2", "生物标志物2", "组织修复和再生能力指标", "-0.179002")
        ]
        
        for biomarker, name, desc, example in biomarkers:
            with st.expander(f"{biomarker}: {name}"):
                st.markdown(f"**描述**: {desc}")
                st.markdown(f"**示例值**: {example}")
                st.markdown(f"**数据精度**: 支持小数点后10位")
        
        st.markdown("### 📈 烧伤类型说明")
        for burn_id, burn_info in burn_type_mapping.items():
            st.markdown(f"""
            <div class="feature-box">
            <strong>{burn_info['cn']}</strong> ({burn_info['en']})
            - <em>描述</em>: {burn_info['description']}
            - <em>颜色标识</em>: <span style="color:{burn_info['color']}">●</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab_guide4:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## 🧠 算法原理")
        
        st.markdown("### 📈 SHAP (SHapley Additive exPlanations) 分析")
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        **理论基础**: 基于博弈论的Shapley值，公平分配每个特征对预测结果的贡献度
        
        **核心原理**:
        - 计算每个特征在所有可能的特征子集中的边际贡献
        - 通过加权平均得到特征的SHAP值
        - 正值表示增加预测概率，负值表示减少预测概率
        
        **数学公式**:
        """)
        st.markdown('<div class="code-box">ϕᵢ = Σ [f(S ∪ {i}) - f(S)] × |S|! × (|F| - |S| - 1)! / |F|!</div>', unsafe_allow_html=True)
        st.markdown("""
        **应用价值**:
        - 理解模型决策依据
        - 识别关键影响因素
        - 提供特征重要性排序
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔗 图网络分析")
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        **理论基础**: 复杂网络理论，将特征视为节点，特征间关系视为边
        
        **核心指标**:
        - **度中心性**: 节点连接数量，反映特征活跃度
        - **介数中心性**: 节点在网络中的桥梁作用
        - **紧密中心性**: 节点到其他节点的平均距离
        
        **网络构建**:
        - 节点: 生物标志物特征
        - 边: 特征间的相关性强度
        - 权重: 基于特征值相似度计算
        
        **应用价值**:
        - 揭示特征间相互作用关系
        - 识别网络中的关键枢纽特征
        - 理解特征系统的整体结构
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔄 反事实分析")
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        **理论基础**: 因果推理，通过改变输入特征观察预测结果变化
        
        **分析方法**:
        - 对每个特征进行微小调整（如±20%、±50%）
        - 观察预测结果的变化
        - 寻找改变预测的最小特征调整
        
        **数学表达**:
        """)
        st.markdown('<div class="code-box">x\' = x + δ → 检查 f(x\') 是否 ≠ f(x)</div>', unsafe_allow_html=True)
        st.markdown("""
        **应用价值**:
        - 提供干预策略建议
        - 理解决策边界
        - 发现模型的敏感特征
        - 为临床干预提供量化依据
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🌳 随机森林算法")
        st.markdown('<div class="theory-box">', unsafe_allow_html=True)
        st.markdown("""
        **算法原理**: 集成学习，通过多个决策树的集体决策提高预测准确性
        
        **核心特点**:
        - **Bagging**: 自助采样构建多个决策树
        - **特征随机性**: 每个节点分裂时随机选择特征子集
        - **投票机制**: 多棵树投票决定最终预测结果
        
        **优势**:
        - 抗过拟合能力强
        - 处理高维数据效果好
        - 提供特征重要性评估
        - 对异常值不敏感
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab_guide5:
        st.markdown('<div class="guide-section">', unsafe_allow_html=True)
        st.markdown("## ❓ 常见问题")
        
        faqs = [
            ("❓ 系统支持的数据格式有哪些？", "支持CSV和Excel格式，需要包含BG1、EGF、IL-1β、BG2四列数据"),
            ("❓ 数据精度可以调整吗？", "支持小数点后6-15位精度，可在系统设置中调整"),
            ("❓ SHAP分析需要多少样本？", "单样本即可进行SHAP分析，多样本可提供更稳定的结果"),
            ("❓ 如何解释反事实分析结果？", "反事实分析显示如何调整特征值来改变预测结果，为干预提供依据"),
            ("❓ 系统支持哪些语言？", "支持中文和英文界面，可在系统设置中切换"),
            ("❓ 分析结果可以导出吗？", "支持导出CSV数据、PNG图表和文本报告")
        ]
        
        for question, answer in faqs:
            with st.expander(question):
                st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "⚙️ 系统设置":
    st.markdown('<div class="main-header">⚙️ 系统设置</div>', unsafe_allow_html=True)
    
    # 语言设置
    st.subheader("🌐 语言设置")
    # 修复：添加唯一的key参数
    language = st.selectbox("选择界面语言", ["中文", "English"], key="system_language_select")
    
    if st.button("💾 应用语言设置", use_container_width=True, key="apply_language_btn"):
        st.session_state.language = language
        st.success("✅ 语言设置已应用")
    
    st.markdown("---")
    
    # 图表颜色设置
    st.subheader("🎨 图表颜色设置")
    
    st.info("当前使用Nature配色方案: #4E79A7, #F28E2B, #E15759, #76B7B2, #59A14F, #EDC948")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color1 = st.color_picker("颜色1", value="#4E79A7", key="color1_picker")
    with col2:
        color2 = st.color_picker("颜色2", value="#F28E2B", key="color2_picker")
    with col3:
        color3 = st.color_picker("颜色3", value="#E15759", key="color3_picker")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        color4 = st.color_picker("颜色4", value="#76B7B2", key="color4_picker")
    with col5:
        color5 = st.color_picker("颜色5", value="#59A14F", key="color5_picker")
    with col6:
        color6 = st.color_picker("颜色6", value="#EDC948", key="color6_picker")
    
    st.markdown("---")
    
    # 图表字体设置
    st.subheader("🔤 图表字体设置")
    
    st.info("设置所有图表中的标题、坐标轴、刻度和标签的字体样式")
    
    col_font1, col_font2 = st.columns(2)
    
    with col_font1:
        st.markdown("#### 标题字体设置")
        # 修复：添加唯一的key参数
        chart_title_family = st.selectbox("标题字体", ["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Times New Roman"], 
                                        key="chart_title_family_select")
        chart_title_size = st.slider("标题字号", 10, 20, 14, key="chart_title_size_slider")
        chart_title_weight = st.selectbox("标题字重", ["normal", "bold"], key="chart_title_weight_select")
        
        st.markdown("#### 坐标轴字体设置")
        chart_axis_family = st.selectbox("坐标轴字体", ["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Times New Roman"], 
                                       key="chart_axis_family_select")
        chart_axis_size = st.slider("坐标轴字号", 8, 16, 10, key="chart_axis_size_slider")
    
    with col_font2:
        st.markdown("#### 刻度字体设置")
        chart_tick_family = st.selectbox("刻度字体", ["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Times New Roman"], 
                                       key="chart_tick_family_select")
        chart_tick_size = st.slider("刻度字号", 6, 14, 8, key="chart_tick_size_slider")
        
        st.markdown("#### 标签字体设置")
        chart_label_family = st.selectbox("标签字体", ["Microsoft YaHei", "SimHei", "SimSun", "Arial", "Times New Roman"], 
                                       key="chart_label_family_select")
        chart_label_size = st.slider("标签字号", 8, 16, 9, key="chart_label_size_slider")
    
    st.markdown("---")
    
    # 数据精度设置
    st.subheader("🔢 数据精度设置")
    
    data_precision_input = st.slider("数据小数点后位数", 6, 15, 10, key="data_precision_slider")
    st.info(f"当前数据精度: 小数点后{data_precision_input}位")
    
    # 主题设置
    st.subheader("🎭 主题设置")
    
    # 修复：添加唯一的key参数
    theme = st.selectbox("选择界面主题", ["浅色主题", "深色主题"], key="theme_select")
    
    # 应用设置按钮
    if st.button("💾 应用所有设置", use_container_width=True, key="apply_all_settings_btn"):
        # 保存设置到session state
        st.session_state.chart_colors = [color1, color2, color3, color4, color5, color6]
        
        # 保存图表字体设置
        st.session_state.chart_title_font = {
            'family': chart_title_family,
            'size': chart_title_size,
            'weight': chart_title_weight
        }
        st.session_state.chart_axis_font = {
            'family': chart_axis_family,
            'size': chart_axis_size
        }
        st.session_state.chart_tick_font = {
            'family': chart_tick_family,
            'size': chart_tick_size
        }
        st.session_state.chart_label_font = {
            'family': chart_label_family,
            'size': chart_label_size
        }
        
        st.session_state.current_data_precision = data_precision_input
        st.session_state.theme = theme
        st.success("✅ 所有设置已应用")
    
    # 重置设置为默认值
    if st.button("🔄 重置为默认设置", use_container_width=True, key="reset_defaults_btn"):
        st.session_state.chart_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
        st.session_state.title_font = {'family': 'Microsoft YaHei', 'size': 14, 'weight': 'bold'}
        st.session_state.label_font = {'family': 'Microsoft YaHei', 'size': 10}
        st.session_state.current_data_precision = 10
        st.session_state.theme = 'light'
        st.success("✅ 已重置为默认设置")
    
    # 当前设置预览
    st.markdown("---")
    st.subheader("📊 当前设置预览")
    
    col_preview1, col_preview2 = st.columns(2)
    
    with col_preview1:
        st.markdown(f"""
        <div class="setting-box">
        <strong>当前设置:</strong>
        <ul>
        <li>语言: {st.session_state.language}</li>
        <li>主题: {st.session_state.theme}</li>
        <li>数据精度: 小数点后{getattr(st.session_state, 'current_data_precision', 10)}位</li>
        <li>字体: {st.session_state.title_font['family']}</li>
        <li>标题字号: {st.session_state.title_font['size']}</li>
        <li>标签字号: {st.session_state.label_font['size']}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_preview2:
        # 颜色预览
        st.markdown("**颜色预览:**")
        colors_html = ""
        for i, color in enumerate(st.session_state.chart_colors):
            colors_html += f'<span style="display: inline-block; width: 20px; height: 20px; background-color: {color}; margin: 2px; border-radius: 3px;" title="颜色{i+1}"></span>'
        st.markdown(f'<div>{colors_html}</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666; font-family: "Microsoft YaHei", sans-serif;">🔥 烧伤智能识别系统 | 基于机器学习的医疗辅助诊断工具 | v1.0 | 本地部署版本</div>', unsafe_allow_html=True)