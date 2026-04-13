import os

# 设置图片所在文件夹路径
folder = "/data/images"   # 修改为你的文件夹路径
output_txt = "/data/list.txt"         # 输出文件名

# 获取文件夹下所有 jpg/png 文件
files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
files.sort()  # 排序，保证顺序一致

# 写入txt文件
with open(output_txt, "w", encoding="utf-8") as f:
    for filename in files:
        f.write(filename + "\n")

print(f"已将 {len(files)} 个文件名写入 {output_txt}")