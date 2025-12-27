from git import Repo


def get_current_commit_hash():
    repo = Repo(".")
    commit_hash = repo.head.commit.hexsha

    return commit_hash
