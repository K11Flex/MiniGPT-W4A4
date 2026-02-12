import requests
import os


def download_and_clip():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    save_path = "data/tinystories.txt"
    limit_mb = 50  # 限制下载 50MB

    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"正在从 Hugging Face 下载并截取前 {limit_mb}MB 数据...")

    # 使用 stream=True 允许我们流式读取，不用等整个 1.8GB 下载完
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 每次读取 1MB
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r已下载: {downloaded / (1024 * 1024):.2f} MB", end="")

                if downloaded >= limit_mb * 1024 * 1024:
                    print("\n达到 50MB 限制，停止下载。")
                    break
    print(f"完成！数据已存至: {save_path}")


if __name__ == "__main__":
    download_and_clip()