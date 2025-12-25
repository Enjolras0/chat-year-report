import json
from datetime import datetime
from collections import Counter
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib

# 设置matplotlib字体以支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取 JSON 数据
with open("chat.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    # 兼容 chat.json 结构
    data = raw_data.get('messages', []) if isinstance(raw_data, dict) else raw_data

# 规范化结构
messages = []
for msg in data:
    # 过滤掉非聊天类型的消息（如系统消息），如果需要的话。
    # 这里保留所有，后续处理。
    
    # 获取时间
    create_time = msg.get('createTime', 0)
    if not create_time:
        continue
        
    dt = datetime.fromtimestamp(create_time)
    
    msg_dict = {
        'time': dt,
        'sender': msg.get('senderDisplayName', '未知'),
        'content': msg.get('content', ''),
        'type': msg.get('type', ''),
        'is_self': msg.get('isSend', 0) == 1
    }
    messages.append(msg_dict)

df = pd.DataFrame(messages)
df = df.sort_values('time') # 确保按时间排序

# 0. 获取史上第一条消息（在过滤之前）
first_msg_ever = None
if not df.empty:
    # 找到第一条非系统消息
    non_sys_msgs = df[df['type'] != '系统消息']
    if not non_sys_msgs.empty:
        first_msg_ever = non_sys_msgs.iloc[0].to_dict()
    else:
        first_msg_ever = df.iloc[0].to_dict()

# 1. 筛选时间范围：2025-01-01 到 2025-12-25
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-12-25 23:59:59")
df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

if df.empty:
    print("指定日期范围内没有聊天记录。")
    exit()

df['hour'] = df['time'].dt.hour
df['date'] = df['time'].dt.date
df['char_count'] = df['content'].apply(len)

# 2. 基础统计
total_messages = len(df)
total_chars = df['char_count'].sum()

# 获取发送者名称（容错处理）
senders = df['sender'].unique()
self_name = "我"
friend_name = "朋友"
for s in senders:
    if df[df['sender'] == s]['is_self'].iloc[0]:
        self_name = s
    else:
        friend_name = s

msg_count_by_person = df['sender'].value_counts()
char_count_by_person = df.groupby('sender')['char_count'].sum()

# 3. 消息类型统计
type_counts = df['type'].value_counts()

# 4. 聊天频率分析（按天）
daily_counts = df.groupby('date').size()
# 补全日期范围（可选，为了图表连续性）
idx = pd.date_range(start_date.date(), end_date.date())
daily_counts = daily_counts.reindex(idx, fill_value=0)

# 5. 活跃时间段分析（按小时）
hourly_distribution = df.groupby('hour').size()
# 补全24小时
hourly_distribution = hourly_distribution.reindex(range(24), fill_value=0)

# 6. 高频词与话题分析
# 定义话题关键词字典
topic_keywords = {
    "🎮 星露谷物语": ["星露谷", "stardew", "Stardew", "鹈鹕镇", "下矿", "种菜", "鱼王", "潘妮", "阿比盖尔", "塞巴斯蒂安", "哈维", "山姆", "亚历克斯", "谢恩", "马鲁", "艾米丽", "海莉", "莱纳斯", "法师", "祝尼魔"],
    "👗 暖暖系列": ["暖暖", "闪暖", "奇迹暖暖", "无限暖暖", "搭配", "套装", "抽阁", "叠纸", "狗叠", "大喵", "秦衣", "左一", "莉莉斯", "墨丘利"],
    "🐱 罗小黑": ["罗小黑", "小黑", "蓝溪镇", "无限", "风息", "老君", "清凝", "玄离", "谛听", "哪吒", "会馆", "灵质空间"],
    "💻 项目开发": ["项目", "代码", "bug", "Bug", "BUG", "开发", "需求", "上线", "数据库", "前端", "后端", "接口", "服务器", "部署", "答辩", "大创", "毕设"],
    "📚 学习上课": ["学习", "上课", "作业", "考试", "复习", "老师", "绩点", "挂科", "考研", "教室", "图书馆", "自习", "早八", "课设", "实验", "论文", "文献"]
}

# 统计话题频次
topic_counts = {k: 0 for k in topic_keywords}
# 记录每个话题下的具体匹配词，用于后续分析（可选）
topic_details = {k: Counter() for k in topic_keywords}

for msg in df['content']:
    if not isinstance(msg, str):
        continue
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in msg:
                topic_counts[topic] += 1
                topic_details[topic][keyword] += 1
                # 一条消息若包含多个关键词，算该话题+1即可（或者每出现一次关键词都加，这里采用消息级计数）
                # 为了避免一条消息多次命中同一话题的不同关键词导致重复计数过多，这里break
                break

# 定义停用词
stop_words = set([
    "啊啊", "哈哈", "哈", "啊", "哦", "嗯", "了", "的", "我", "你", "是", "在", "不", "有", "也", "就", "都",
    "吧", "吗", "呢", "去", "要", "这", "那", "个", "很", "好", "么", "怎么", "什么", "因为", "所以",
    "但是", "而且", "然后", "虽然", "其实", "就是", "还是", "或者", "如果", "那个", "这个", "那个",
    "一个", "这么", "我们", "没有", "知道", "时候", "特别", "不是", "这样", "觉得", "感觉", "真的", "现在", "可以", "自己", "可能", "还有", "那些", "这些", "一次", "一下", "一点", "一些",
    "[动画表情]", "[图片]", "[语音]", "[视频]", "[引用]", "[链接]", "[文件]", "[位置]", "[转账]", 
    "拍了拍", "emoji", "表情", "ok", "OK", "Ok", "xxx", "哈哈哈", "啊啊啊", "嘿嘿", "嘻嘻", "呜呜", "emmm",
    "捂脸", "流泪", "抓狂", "憨笑", "拥抱", "呲牙", "偷笑", "调皮", "撇嘴", "发呆"
])

text_df = df[df['type'] == '文本消息']
tokens = []
all_text = ""

for msg in text_df['content']:
    if not isinstance(msg, str):
        continue
    words = jieba.lcut(msg)
    for w in words:
        if len(w) > 1 and w not in stop_words and not w.startswith('[') and not w.isnumeric():
             tokens.append(w)

word_freq = Counter(tokens).most_common(100)

# 7. 生成图表

# 每日聊天频率趋势图
plt.figure(figsize=(12, 5))
plt.plot(daily_counts.index, daily_counts.values, color='#ff9999', linewidth=2)
plt.title(f"每日聊天频率 ({start_date.date()} - {end_date.date()})")
plt.xlabel("日期")
plt.ylabel("消息数")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("daily_trend.png")
plt.close()

# 活跃时间段图
plt.figure(figsize=(10, 5))
hourly_distribution.plot(kind='bar', color='skyblue', width=0.8)
plt.title("活跃时间段（按小时）")
plt.xlabel("小时 (0-23)")
plt.ylabel("消息数")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("hourly_activity.png")
plt.close()

# 词云生成
if tokens:
    try:
        wc = WordCloud(
            font_path='msyh.ttc', 
            background_color='white', 
            width=1000, 
            height=800,
            stopwords=stop_words,
            collocations=False
        )
        wc.generate_from_frequencies(dict(word_freq))
        wc.to_file("wordcloud.png")
    except Exception as e:
        print(f"生成词云失败 (可能是字体路径问题): {e}")
        
# 话题分布图
plt.figure(figsize=(10, 6))
# 过滤掉计数为0的话题（可选）
filtered_topics = {k: v for k, v in topic_counts.items() if v > 0}
if filtered_topics:
    # 排序
    sorted_topics = dict(sorted(filtered_topics.items(), key=lambda item: item[1], reverse=True))
    plt.bar(sorted_topics.keys(), sorted_topics.values(), color=['#FF9999', '#66B2FF', '#99CC99', '#FFCC99', '#CC99FF'])
    plt.title("话题热度分析")
    plt.xlabel("话题")
    plt.ylabel("相关消息数")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    # 在柱状图上显示数值
    for i, v in enumerate(sorted_topics.values()):
        plt.text(i, v + max(sorted_topics.values())*0.01, str(v), ha='center')
    plt.tight_layout()
    plt.savefig("topic_distribution.png")
    plt.close()

# 8. 生成年度报告 Markdown
report_file = "chat_year_report.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(f"# � 2025 年度聊天报告\n\n")
    f.write(f"> 记录时间：{start_date.date()} 至 {end_date.date()}\n\n")
    
    f.write("## 📊 基础概览\n")
    f.write(f"- **总消息数**：{total_messages}\n")
    f.write(f"- **总字数**：{total_chars}\n")
    f.write(f"- **日均消息**：{total_messages / len(daily_counts):.1f}\n\n")
    
    f.write("## 👥 谁是话痨？\n")
    f.write("| 昵称 | 消息数 | 字数 |\n")
    f.write("| --- | --- | --- |\n")
    for sender in msg_count_by_person.index:
        count = msg_count_by_person[sender]
        chars = char_count_by_person.get(sender, 0)
        f.write(f"| {sender} | {count} | {chars} |\n")
    f.write("\n")
    
    f.write("## 📈 聊天频率分析\n")
    f.write("### 每日趋势\n")
    f.write("![每日趋势](daily_trend.png)\n\n")
    f.write("### 活跃时间段\n")
    f.write("![活跃时间](hourly_activity.png)\n\n")
    
    f.write("## 🗣 高频话题与热词\n")
    f.write("### 📌 话题热度排行\n")
    f.write("![话题分布](topic_distribution.png)\n\n")
    
    # 输出话题详情
    for topic, count in sorted_topics.items():
        if count > 0:
            f.write(f"#### {topic} (共 {count} 条)\n")
            # 展示该话题下最高频的3个关键词
            top_keywords = topic_details[topic].most_common(5)
            keyword_str = "、".join([f"{k}({v})" for k, v in top_keywords])
            f.write(f"> 关键词：{keyword_str}\n\n")

    f.write("![词云](wordcloud.png)\n\n")
    f.write("### 🔥 Top 20 热词\n")
    for i, (word, freq) in enumerate(word_freq[:20], 1):
        f.write(f"{i}. **{word}** ({freq})\n")

# 9. 生成 HTML 年度报告
def generate_html_report():
    html_file = "year_report.html"
    
    # 准备数据
    # 每日数据: [date_str, count]
    daily_data = [[d.strftime('%Y-%m-%d'), int(c)] for d, c in daily_counts.items()]
    
    # 活跃时段: [hour, count]
    hourly_data = [int(c) for c in hourly_distribution.values]
    
    # 话题数据: [{'name': topic, 'value': count}]
    topic_data = [{'name': k, 'value': v} for k, v in sorted_topics.items()]
    
    # 词云数据: [{'name': word, 'value': freq}]
    word_cloud_data = [{'name': w, 'value': f} for w, f in word_freq]
    
    # 发言对比
    sender_data = []
    for sender in msg_count_by_person.index:
        sender_data.append({
            'name': sender, 
            'value': int(msg_count_by_person[sender]),
            'chars': int(char_count_by_person.get(sender, 0))
        })

    # 消息类型数据
    type_data = [{'name': k, 'value': int(v)} for k, v in type_counts.items()]

    # 第一条消息数据
    first_msg_2025 = df.iloc[0].to_dict() if not df.empty else None
    
    # 格式化消息内容（处理非文本消息）
    def format_content(msg):
        if not msg: return "无内容"
        content = msg['content']
        msg_type = msg['type']
        if msg_type != '文本消息':
            return f"[{msg_type}]"
        return content

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>2025 年度聊天报告</title>
    <!-- Swiper CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css" />
    <!-- Animate.css -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #f0f2f5;
            font-family: 'Microsoft YaHei', sans-serif;
            overflow: hidden; /* Prevent native scroll */
        }}
        .swiper {{
            width: 100vw;
            height: 100vh;
        }}
        .swiper-slide {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #fff;
            box-sizing: border-box;
            padding: 20px;
            overflow: hidden;
            position: relative;
        }}
        
        /* Custom Slide Styles */
        .slide-cover {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
        }}
        .slide-cover h1 {{ font-size: 2.2em; margin-bottom: 10px; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
        .slide-cover p {{ font-size: 1.1em; opacity: 0.9; }}
        
        .slide-title {{
            font-size: 1.4em;
            color: #764ba2;
            margin-bottom: 15px;
            font-weight: bold;
            text-align: center;
            width: 100%;
            z-index: 10;
        }}
        
        .chart-container {{
            width: 100%;
            height: 45vh;
            min-height: 250px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            width: 100%;
            margin-bottom: 20px;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .stat-val {{ font-size: 1.5em; color: #764ba2; font-weight: bold; }}
        .stat-lbl {{ color: #666; font-size: 0.8em; }}
        
        .memory-box {{
            width: 100%;
            background: #fff0f5;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
            border: 1px solid #ffdeeb;
            font-size: 0.9em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .memory-time {{ color: #d63384; font-weight: bold; font-size: 0.8em; margin-bottom: 5px;}}
        .memory-content {{ 
            background: white; 
            padding: 8px; 
            border-radius: 5px; 
            border-left: 3px solid #d63384; 
            word-break: break-all;
            max-height: 100px;
            overflow-y: auto;
        }}

        /* Animation hint */
        .swipe-hint {{
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            animation: bounce 2s infinite;
            font-size: 0.9em;
            opacity: 0.8;
            z-index: 100;
        }}
        
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{transform: translateX(-50%) translateY(0);}}
            40% {{transform: translateX(-50%) translateY(-10px);}}
            60% {{transform: translateX(-50%) translateY(-5px);}}
        }}
        
        /* Swiper Pagination Customization */
        .swiper-pagination-bullet-active {{
            background: #764ba2 !important;
        }}
    </style>
</head>
<body>
    <div class="swiper mySwiper">
        <div class="swiper-wrapper">
            <!-- Slide 1: Cover -->
            <div class="swiper-slide slide-cover">
                <div class="animate__animated animate__fadeInDown">
                    <h1>📅 2025<br>年度聊天报告</h1>
                    <p>{start_date.date()} ~ {end_date.date()}</p>
                    <div style="margin-top: 40px; font-size: 3em;">🎁</div>
                </div>
                <div class="swipe-hint">☝️ 上滑开启回忆</div>
            </div>
            
            <!-- Slide 2: Overview & Memory -->
            <div class="swiper-slide">
                <div class="slide-title animate__animated animate__fadeInLeft">🌟 我们的回忆</div>
                
                <div class="stats-grid animate__animated animate__zoomIn">
                    <div class="stat-item">
                        <div class="stat-val">{total_messages}</div>
                        <div class="stat-lbl">总消息数</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val">{len(daily_counts)}</div>
                        <div class="stat-lbl">聊天天数</div>
                    </div>
                </div>
                
                <div class="memory-box animate__animated animate__fadeInUp" style="animation-delay: 0.2s;">
                    <div>🚀 <strong>2025 第一声问候</strong></div>
                    <div class="memory-time">{first_msg_2025['time'].strftime('%Y-%m-%d %H:%M:%S') if first_msg_2025 else '无'}</div>
                    <div class="memory-content">
                        <strong>{first_msg_2025['sender'] if first_msg_2025 else ''}:</strong>
                        {format_content(first_msg_2025)}
                    </div>
                </div>
                
                 <div class="memory-box animate__animated animate__fadeInUp" style="animation-delay: 0.4s;">
                    <div>�️ <strong>最初的相遇</strong></div>
                    <div class="memory-time">{first_msg_ever['time'].strftime('%Y-%m-%d %H:%M:%S') if first_msg_ever else '无'}</div>
                    <div class="memory-content">
                         <strong>{first_msg_ever['sender'] if first_msg_ever else ''}:</strong>
                         {format_content(first_msg_ever)}
                    </div>
                </div>
            </div>
            
            <!-- Slide 3: Sender & Type -->
            <div class="swiper-slide">
                <div class="slide-title">👥 谁更爱说话？</div>
                <div id="senderChart" class="chart-container" style="height: 30vh;"></div>
                <div class="slide-title" style="margin-top: 15px; font-size: 1.2em;">📨 消息类型</div>
                <div id="typeChart" class="chart-container" style="height: 30vh;"></div>
            </div>
            
            <!-- Slide 4: Daily Trend -->
            <div class="swiper-slide">
                <div class="slide-title">📈 这一年的起伏</div>
                <div id="dailyChart" class="chart-container" style="height: 60vh;"></div>
            </div>
            
            <!-- Slide 5: Hourly Activity -->
            <div class="swiper-slide">
                <div class="slide-title">⏰ 我们什么时候最活跃？</div>
                <div id="hourlyChart" class="chart-container" style="height: 60vh;"></div>
            </div>
            
            <!-- Slide 6: Topics -->
            <div class="swiper-slide">
                <div class="slide-title">🗣 我们最爱聊...</div>
                <div id="topicChart" class="chart-container" style="height: 65vh;"></div>
            </div>
            
            <!-- Slide 7: WordCloud -->
            <div class="swiper-slide">
                <div class="slide-title">🌈 年度关键词</div>
                <div id="wordCloudChart" class="chart-container" style="height: 60vh;"></div>
            </div>
            
            <!-- Slide 8: End -->
            <div class="swiper-slide slide-cover" style="background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);">
                <div class="animate__animated animate__zoomIn">
                    <h1 style="font-size: 4em;">❤️</h1>
                    <h2>感谢有你</h2>
                    <p style="margin-top: 20px;">2026，未完待续...</p>
                </div>
            </div>
        </div>
        <!-- Pagination -->
        <div class="swiper-pagination"></div>
    </div>

    <!-- Swiper JS -->
    <script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>
    
    <script>
        // Data Injection
        const dailyData = {daily_data};
        const hourlyData = {hourly_data};
        const topicData = {topic_data};
        const wordCloudData = {word_cloud_data};
        const senderData = {sender_data};
        const typeData = {type_data};

        // Init Swiper
        var swiper = new Swiper(".mySwiper", {{
            direction: "vertical",
            pagination: {{
                el: ".swiper-pagination",
                clickable: true,
            }},
            mousewheel: true,
            effect: 'slide',
            on: {{
                slideChangeTransitionEnd: function () {{
                    resizeCharts();
                }}
            }}
        }});

        // Chart Initialization
        const senderChart = echarts.init(document.getElementById('senderChart'));
        const typeChart = echarts.init(document.getElementById('typeChart'));
        const dailyChart = echarts.init(document.getElementById('dailyChart'));
        const hourlyChart = echarts.init(document.getElementById('hourlyChart'));
        const topicChart = echarts.init(document.getElementById('topicChart'));
        const wordCloudChart = echarts.init(document.getElementById('wordCloudChart'));
        
        const charts = [senderChart, typeChart, dailyChart, hourlyChart, topicChart, wordCloudChart];
        
        function resizeCharts() {{
            charts.forEach(chart => chart.resize());
        }}
        
        window.addEventListener('resize', resizeCharts);

        // --- Chart Options ---
        
        senderChart.setOption({{
            tooltip: {{ trigger: 'item' }},
            legend: {{ bottom: '0%', left: 'center' }},
            series: [{{
                name: '消息数',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '45%'],
                itemStyle: {{ borderRadius: 8, borderColor: '#fff', borderWidth: 2 }},
                data: senderData
            }}]
        }});
        
        typeChart.setOption({{
            tooltip: {{ trigger: 'item' }},
            legend: {{ bottom: '0%', left: 'center' }},
            series: [{{
                name: '类型',
                type: 'pie',
                radius: '60%',
                center: ['50%', '45%'],
                data: typeData
            }}]
        }});
        
        dailyChart.setOption({{
            grid: {{ left: '3%', right: '5%', bottom: '10%', top: '10%', containLabel: true }},
            tooltip: {{ trigger: 'axis' }},
            xAxis: {{ type: 'category', data: dailyData.map(i=>i[0]) }},
            yAxis: {{ type: 'value' }},
            series: [{{
                data: dailyData.map(i=>i[1]),
                type: 'line',
                smooth: true,
                areaStyle: {{ opacity: 0.3 }},
                itemStyle: {{ color: '#764ba2' }}
            }}]
        }});
        
        hourlyChart.setOption({{
            grid: {{ left: '3%', right: '5%', bottom: '10%', top: '10%', containLabel: true }},
            tooltip: {{ trigger: 'axis' }},
            xAxis: {{ type: 'category', data: Array.from({{length:24}},(_,i)=>i+'点') }},
            yAxis: {{ type: 'value' }},
            series: [{{
                data: hourlyData,
                type: 'bar',
                itemStyle: {{ color: new echarts.graphic.LinearGradient(0,0,0,1,[{{offset:0,color:'#83bff6'}},{{offset:1,color:'#188df0'}}]) }}
            }}]
        }});
        
        topicChart.setOption({{
            grid: {{ left: '3%', right: '8%', bottom: '3%', top: '5%', containLabel: true }},
            tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
            xAxis: {{ type: 'value' }},
            yAxis: {{ type: 'category', data: topicData.map(i=>i.name).reverse() }},
            series: [{{
                data: topicData.map(i=>i.value).reverse(),
                type: 'bar',
                label: {{ show: true, position: 'right' }},
                itemStyle: {{ color: '#ff9999' }}
            }}]
        }});
        
        wordCloudChart.setOption({{
            series: [{{
                type: 'wordCloud',
                shape: 'circle',
                left: 'center', top: 'center',
                width: '100%', height: '100%',
                right: 0, bottom: 0,
                sizeRange: [12, 60],
                rotationRange: [-45, 45],
                gridSize: 8,
                drawOutOfBound: false,
                textStyle: {{
                    fontFamily: 'sans-serif',
                    fontWeight: 'bold',
                    color: function () {{
                        return 'rgb(' + [
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160),
                            Math.round(Math.random() * 160)
                        ].join(',') + ')';
                    }}
                }},
                emphasis: {{ focus: 'self', textStyle: {{ shadowBlur: 10, shadowColor: '#333' }} }},
                data: wordCloudData
            }}]
        }});
        
        // Initial resize
        setTimeout(resizeCharts, 500);
    </script>
</body>
</html>
    """
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"H5网页报告已生成：{html_file}")

generate_html_report()

