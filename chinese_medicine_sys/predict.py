import json
import requests
import sys
import csv
import re
from openai import OpenAI
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import jieba
from collections import Counter
import hashlib
from functools import lru_cache

# 读取训练数据
train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
symptoms_data = []  # 症状
ZX = []  # 证型
ZF = []  # 治法

with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        symptoms_data.append(row[1])  # 提取症状列
        ZX.append(row[2])  # 提取证型列
        ZF.append(row[3])  # 提取治法列

# 配置大模型API参数
client = OpenAI(
    api_key="dab88b90d1466275d34b5af41eab74d4aff5768d",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3"
)

# 全局变量
vectorizer = None
symptom_vectors = None
candidate_zx = None
candidate_zf = None

def initialize_components():
    """初始化所有组件"""
    global vectorizer, symptom_vectors, candidate_zx, candidate_zf
    
    # 初始化候选集合
    candidate_zx = list(set(ZX))
    candidate_zf = list(set(ZF))
    
    # 初始化向量器
    vectorizer, symptom_vectors = build_symptom_embedding()
    
    print(f"初始化完成: 候选证型 {len(candidate_zx)} 种, 候选治法 {len(candidate_zf)} 种")

def preprocess_symptoms_data():
    """数据预处理"""
    symptom_keywords = []
    for symptom in symptoms_data:
        words = jieba.cut(symptom)
        keywords = [word for word in words if len(word) > 1]  # 过滤单字
        symptom_keywords.append(" ".join(keywords))
    return symptom_keywords

def build_symptom_embedding():
    """构建症状向量表示"""
    symptom_keywords = preprocess_symptoms_data()
    vectorizer = TfidfVectorizer()
    symptom_vectors = vectorizer.fit_transform(symptom_keywords)
    return vectorizer, symptom_vectors

def find_most_similar_examples(input_symptom, top_k=3):
    """找到最相似的训练样本"""
    try:
        input_keywords = " ".join([word for word in jieba.cut(input_symptom) if len(word) > 1])
        input_vector = vectorizer.transform([input_keywords])
        similarities = cosine_similarity(input_vector, symptom_vectors)
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        
        similar_examples = []
        for idx in top_indices:
            similar_examples.append({
                "症状": symptoms_data[idx],
                "证型": ZX[idx],
                "治法": ZF[idx],
                "相似度": float(similarities[0][idx])
            })
        
        return similar_examples
    except Exception as e:
        print(f"相似度计算异常: {str(e)}")
        # 返回默认的前几个示例
        return [{"症状": symptoms_data[i], "证型": ZX[i], "治法": ZF[i], "相似度": 0.5} for i in range(min(3, len(symptoms_data)))]

def parse_response(content):
    """
    解析大模型返回的响应内容
    处理包含代码块的JSON响应
    """
    if not content:
        return None
        
    # 检查是否包含代码块
    code_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = code_block_pattern.search(content)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {json_str}")
    else:
        # 尝试直接解析整个内容
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取可能的JSON部分
            json_pattern = re.compile(r'\{.*?"证型".*?"治法".*?\}', re.DOTALL)
            match = json_pattern.search(content)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            print(f"直接JSON解析失败: {content}")
    
    return None

def get_fallback_from_similar(similar_examples):
    """从相似病例中获取回退结果"""
    if similar_examples:
        return {
            "证型": similar_examples[0]["证型"],
            "治法": similar_examples[0]["治法"],
            "置信度": "基于相似病例"
        }
    else:
        return {"证型": candidate_zx[0], "治法": candidate_zf[0], "置信度": "默认"}

def validate_and_refine_result(result, similar_examples):
    """结果验证和精炼"""
    if not result or "证型" not in result or "治法" not in result:
        return get_fallback_from_similar(similar_examples)
    
    if result["证型"] not in candidate_zx or result["治法"] not in candidate_zf:
        print(f"结果不在候选范围内: {result}")
        return get_fallback_from_similar(similar_examples)
    
    # 添加置信度评估
    if similar_examples and similar_examples[0]["相似度"] > 0.7:
        result["置信度"] = "高"
    elif similar_examples and similar_examples[0]["相似度"] > 0.4:
        result["置信度"] = "中"
    else:
        result["置信度"] = "低"
    
    return result

def enhanced_call_large_model(symptoms):
    """
    增强版：相似度匹配 + 大模型推理
    """
    # 找到最相似的示例
    similar_examples = find_most_similar_examples(symptoms)
    
    # 构建示例文本
    examples_text = "\n".join([
        f"示例{i+1}: 症状「{ex['症状']}」→ 证型「{ex['证型']}」→ 治法「{ex['治法']}」(相似度:{ex['相似度']:.2f})"
        for i, ex in enumerate(similar_examples)
    ])
    
    system_prompt = f"""
你是资深中医专家，基于相似病例进行证型治法判断。

最相关的参考病例：
{examples_text}

候选证型：{candidate_zx}
候选治法：{candidate_zf}

判断规则：
1. 仔细分析症状与参考病例的相似性
2. 只能从上述候选集合中选择证型和治法
3. 证型与治法必须逻辑对应
4. 参考相似病例的判断逻辑

请严格按照以下JSON格式输出：
{{"证型": "选择的证型", "治法": "选择的治法"}}
"""
    
    user_prompt = f"""
请分析以下症状，参考相似病例进行辨证：

症状：{symptoms}

请输出JSON结果：
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=512,
            temperature=0.1,
            top_p=0.8
        )
        
        content = response.choices[0].message.content
        result = parse_response(content)
        
        return validate_and_refine_result(result, similar_examples)
        
    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return get_fallback_from_similar(similar_examples)

def chain_of_thought_prediction(symptoms):
    """
    思维链推理：让模型分步思考（高精度版本）
    """
    similar_examples = find_most_similar_examples(symptoms)
    examples_text = "\n".join([
        f"症状：「{ex['症状']}」→ 证型：「{ex['证型']}」，治法：「{ex['治法']}」"
        for ex in similar_examples[:2]  # 只使用前2个最相似的
    ])
    
    system_prompt = f"""
你是经验丰富的中医专家，请按步骤推理：

参考病例：
{examples_text}

候选证型：{candidate_zx}
候选治法：{candidate_zf}

推理步骤：
1. 分析症状中的关键辨证要素（如气虚、痰湿、血瘀等）
2. 对比参考病例，找出相似模式
3. 选择最匹配的证型
4. 确定对应的治法
5. 验证证型-治法的逻辑一致性

请先思考分析，然后输出最终结果。
"""
    
    user_prompt = f"""
请分析以下症状：

症状：{symptoms}

请先进行思考分析（不要输出JSON）：
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # 第一轮：获取思考过程
        response1 = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=300,
            temperature=0.1
        )
        
        thinking = response1.choices[0].message.content
        
        # 第二轮：基于思考结果输出最终答案
        follow_up = f"""
基于你的分析，请输出最终的证型和治法（仅输出JSON格式）：

请输出：
{{"证型": "证型名称", "治法": "治法名称"}}
"""
        
        messages.append({"role": "assistant", "content": thinking})
        messages.append({"role": "user", "content": follow_up})
        
        response2 = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=200,
            temperature=0.1
        )
        
        content = response2.choices[0].message.content
        result = parse_response(content)
        
        return validate_and_refine_result(result, similar_examples)
        
    except Exception as e:
        print(f"思维链推理异常: {str(e)}")
        return get_fallback_from_similar(similar_examples)

def simple_call_large_model(symptoms):
    """
    简单快速版本（适合实时推理）
    """
    # 构建少量示例
    examples = []
    for i in range(min(3, len(symptoms_data))):
        examples.append(f"症状：{symptoms_data[i]} → 证型：{ZX[i]}，治法：{ZF[i]}")
    
    examples_text = "\n".join(examples)
    
    system_prompt = f"""
你是中医专家，根据症状判断证型和治法。

学习示例：
{examples_text}

候选证型：{candidate_zx}
候选治法：{candidate_zf}

要求：从候选集合中选择，输出JSON格式。
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"症状：{symptoms}"}
    ]

    try:
        response = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=200,
            temperature=0.1
        )

        content = response.choices[0].message.content
        result = parse_response(content)
        
        similar_examples = find_most_similar_examples(symptoms)
        return validate_and_refine_result(result, similar_examples)

    except Exception as e:
        print(f"API调用异常: {str(e)}")
        similar_examples = find_most_similar_examples(symptoms)
        return get_fallback_from_similar(similar_examples)

def ensemble_prediction(symptoms):
    """
    集成预测：多种方法投票（高精度版本）
    """
    # 方法1：增强版大模型预测
    result1 = enhanced_call_large_model(symptoms)
    
    # 方法2：思维链推理
    result2 = chain_of_thought_prediction(symptoms)
    
    # 方法3：简单快速版本
    result3 = simple_call_large_model(symptoms)
    
    # 方法4：相似度匹配（直接使用最相似的）
    similar_examples = find_most_similar_examples(symptoms)
    result4 = get_fallback_from_similar(similar_examples)
    
    # 投票机制
    predictions = [result1, result2, result3, result4]
    zx_predictions = [pred.get("证型", "未知") for pred in predictions]
    zf_predictions = [pred.get("治法", "待定") for pred in predictions]
    
    # 选择最频繁的预测
    final_zx = Counter(zx_predictions).most_common(1)[0][0]
    final_zf = Counter(zf_predictions).most_common(1)[0][0]
    
    # 如果最高频次相同，选择置信度高的
    zx_counts = Counter(zx_predictions)
    zf_counts = Counter(zf_predictions)
    
    # 检查是否有多个相同频次
    if len([count for count in zx_counts.values() if count == zx_counts[final_zx]]) > 1:
        # 选择置信度高的
        confidence_scores = {}
        for pred in predictions:
            zx = pred.get("证型")
            confidence = 1 if pred.get("置信度") == "高" else 0.5 if pred.get("置信度") == "中" else 0
            confidence_scores[zx] = confidence_scores.get(zx, 0) + confidence
        
        final_zx = max(confidence_scores.items(), key=lambda x: x[1])[0]
    
    return {
        "证型": final_zx,
        "治法": final_zf,
        "投票详情": {
            "增强版": result1,
            "思维链": result2,
            "快速版": result3,
            "相似度": result4
        }
    }

# 缓存机制
@lru_cache(maxsize=500)
def cached_prediction(symptom_hash):
    """缓存预测结果"""
    symptom = [s for s in symptoms_data if hashlib.md5(s.encode()).hexdigest() == symptom_hash][0]
    return enhanced_call_large_model(symptom)

def predict(symptom):
    """
    主预测函数
    
    Parameters:
    -----------
    symptom : str
        症状描述
    mode : str
        预测模式："fast"快速模式, "balanced"平衡模式, "accurate"高精度模式
    
    Returns:
    --------
    zx_predict, zf_predict : str
        证型预测结果，治法预测结果
    """
    mode="accurate"


    # 初始化组件（如果未初始化）
    if vectorizer is None:
        initialize_components()
    
    if mode == "fast":
        # 快速模式：使用简单版本
        result = simple_call_large_model(symptom)
    elif mode == "accurate":
        # 高精度模式：使用集成学习
        result = ensemble_prediction(symptom)
    else:
        # 平衡模式：使用增强版本
        result = enhanced_call_large_model(symptom)
    
    zx_predict = result.get('证型', candidate_zx[0] if candidate_zx else '未知')
    zf_predict = result.get('治法', candidate_zf[0] if candidate_zf else '待定')
    
    return zx_predict, zf_predict
