import pymysql
import os
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 配置 MySQL 连接
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',
#     'password': 'password',
#     'database': 'wedding',
#     'charset': 'utf8mb4',
#     'cursorclass': pymysql.cursors.DictCursor
# }

# 从环境变量获取 Railway 数据库信息（推荐方式）
DB_CONFIG = {
    'host': os.environ.get('RAILWAY_MYSQL_HOST', '默认值'),  # 如 containers-us-west-199.railway.app
    'user': os.environ.get('RAILWAY_MYSQL_USER', '默认值'),  # 自动生成的用户名
    'password': os.environ.get('RAILWAY_MYSQL_PASSWORD', '默认值'),  # 自动生成的密码
    'database': os.environ.get('RAILWAY_MYSQL_DB', 'wedding'),  # 数据库名（默认可能为 railway）
    # 'port': int(os.environ.get('RAILWAY_MYSQL_PORT', 3306)),  # 端口号（如 3306）
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 或直接使用 Railway 提供的完整连接字符串
# DATABASE_URL = os.environ.get('DATABASE_URL')


# 初始化数据库
def init_db():
    conn = pymysql.connect(**DB_CONFIG)
    with conn.cursor() as cursor:
        cursor.execute('''CREATE TABLE IF NOT EXISTS rsvp (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            attend VARCHAR(10) NOT NULL,
            participant_count INT,
            contact_phone VARCHAR(20)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS comments (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            comment TEXT NOT NULL
        )''')
    conn.commit()
    conn.close()


init_db()


@app.route('/rsvp', methods=['POST'])
def rsvp():
    try:
        data = request.get_json()
        name = data.get('name')
        attend = data.get('attend')
        participant_count = data.get('participant_count')
        contact_phone = data.get('contact_phone')
        if not name or not attend or not participant_count or not contact_phone:
            return jsonify({"message": "请填写完整信息"}), 400

        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = "INSERT INTO rsvp (name, attend, participant_count, contact_phone) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (name, attend, participant_count, contact_phone))
        conn.commit()
        conn.close()
        return jsonify({"message": "RSVP 提交成功"}), 200
    except Exception as e:
        return jsonify({"message": f"提交失败: {str(e)}"}), 400


@app.route('/comment', methods=['POST'])
def comment():
    try:
        data = request.get_json()
        name = data.get('name')
        comment = data.get('comment')
        if not name or not comment:
            return jsonify({"message": "请填写完整信息"}), 400

        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = "INSERT INTO comments (name, comment) VALUES (%s, %s)"
            cursor.execute(sql, (name, comment))
        conn.commit()
        conn.close()
        return jsonify({"message": "评论提交成功"}), 200
    except Exception as e:
        return jsonify({"message": f"评论提交失败: {str(e)}"}), 400


@app.route('/comments', methods=['GET'])
def get_comments():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = "SELECT name, comment FROM comments"
            cursor.execute(sql)
            comments = cursor.fetchall()
        conn.close()
        return jsonify(comments), 200
    except Exception as e:
        return jsonify({"message": f"获取评论失败: {str(e)}"}), 400


@app.route('/all_data', methods=['GET'])
def get_all_data():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = "SELECT id, name, attend, participant_count, contact_phone FROM rsvp"
            cursor.execute(sql)
            rsvp_data = cursor.fetchall()

            sql = "SELECT id, name, comment FROM comments"
            cursor.execute(sql)
            comment_data = cursor.fetchall()
        conn.close()
        return jsonify({"rsvp": rsvp_data, "comments": comment_data}), 200
    except Exception as e:
        return jsonify({"message": f"获取所有数据失败: {str(e)}"}), 400


@app.route('/data_view', methods=['GET'])
def data_view():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            sql = "SELECT id, name, attend, participant_count, contact_phone FROM rsvp"
            cursor.execute(sql)
            rsvp_data = cursor.fetchall()

            sql = "SELECT id, name, comment FROM comments"
            cursor.execute(sql)
            comment_data = cursor.fetchall()
        conn.close()

        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>婚礼数据管理</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>RSVP 数据</h1>
    <table>
        <tr>
            <th>ID</th>
            <th>姓名</th>
            <th>是否出席</th>
            <th>参与人数</th>
            <th>联系电话</th>
        </tr>
        {% for item in rsvp_data %}
        <tr>
            <td>{{ item.id }}</td>
            <td>{{ item.name }}</td>
            <td>{{ item.attend }}</td>
            <td>{{ item.participant_count }}</td>
            <td>{{ item.contact_phone }}</td>
        </tr>
        {% endfor %}
    </table>

    <h1>评论数据</h1>
    <table>
        <tr>
            <th>ID</th>
            <th>姓名</th>
            <th>评论</th>
        </tr>
        {% for item in comment_data %}
        <tr>
            <td>{{ item.id }}</td>
            <td>{{ item.name }}</td>
            <td>{{ item.comment }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
        """
        return render_template_string(html_template, rsvp_data=rsvp_data, comment_data=comment_data)
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # 使用不同端口避免冲突