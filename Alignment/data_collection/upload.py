from huggingface_hub import HfApi, HfFolder

# 登录 Hugging Face
HfFolder.save_token('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')

# 创建 API 实例
api = HfApi()

for i in range(10):
    # 创建仓库
    repo_id = f'yifangong/split_candidate_dataset_part_{i}'
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=False)

    # 上传 JSONL 文件
    api.upload_file(
        path_or_fileobj=f'./candidate_dataset/data_{i}.jsonl',
        path_in_repo='data.jsonl',
        repo_id=repo_id,
        repo_type='dataset'
    )
