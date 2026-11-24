import base64
import os
import subprocess
import sys
import time
import webbrowser
import requests

CLIENT_ID = "178c6fc778ccc68e1d6a"  # Official GitHub CLI OAuth Client ID
REPO_NAME = "Bluer-Girl-Pro"
REPO_DESC = "🛡️ Bluer Girl Pro v6.0 — أداة ذكاء اصطناعي فورية وخفيفة للستريمرز والشاشة ومتصفح كروم لتمويه الأجزاء المكشوفة في الفيديوهات والألعاب (قراند) والأنمي"


def copy_to_clipboard(text: str):
    try:
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
        p.communicate(input=text.encode("utf-8"))
    except Exception:
        pass


def get_github_token() -> str:
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gh_token")
    if os.path.exists(token_file):
        saved = open(token_file, "r", encoding="utf-8").read().strip()
        if saved:
            r = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {saved}"},
                timeout=10,
            )
            if r.status_code == 200:
                return saved

    resp = requests.post(
        "https://github.com/login/device/code",
        headers={"Accept": "application/json"},
        data={"client_id": CLIENT_ID, "scope": "repo"},
        timeout=15,
    ).json()

    device_code = resp["device_code"]
    user_code = resp["user_code"]
    verify_url = resp["verification_uri"]
    interval = int(resp.get("interval", 5))

    copy_to_clipboard(user_code)
    print("=" * 65, flush=True)
    print(f"🔑 كود التفعيل لحسابك في GitHub (تم نسخه للحافظة تلقائياً): {user_code}", flush=True)
    print(f"🌐 افتح الرابط وحط الكود واضغط Authorize: {verify_url}", flush=True)
    print("=" * 65, flush=True)

    try:
        webbrowser.open(verify_url)
    except Exception:
        pass

    while True:
        time.sleep(interval)
        poll = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=15,
        ).json()

        if "access_token" in poll:
            token = poll["access_token"]
            try:
                open(token_file, "w", encoding="utf-8").write(token)
            except Exception:
                pass
            return token

        err = poll.get("error")
        if err == "authorization_pending":
            continue
        elif err == "slow_down":
            interval += 5
        else:
            raise RuntimeError(f"GitHub authorization stopped: {err}")


def upload_project():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    token = get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    user_info = requests.get(
        "https://api.github.com/user", headers=headers, timeout=15
    ).json()
    username = user_info["login"]
    print(f"\n✅ تم الدخول بحسابك في GitHub: @{username}", flush=True)

    # 1. Create or check repository
    repo_url = f"https://api.github.com/repos/{username}/{REPO_NAME}"
    r_check = requests.get(repo_url, headers=headers, timeout=15)
    if r_check.status_code == 404:
        print(f"📦 جاري إنشاء المستودع {REPO_NAME} في حسابك...", flush=True)
        r_create = requests.post(
            "https://api.github.com/user/repos",
            headers=headers,
            json={
                "name": REPO_NAME,
                "description": REPO_DESC,
                "private": False,
                "auto_init": True,
            },
            timeout=20,
        )
        r_create.raise_for_status()
        time.sleep(2)
    else:
        print(f"📦 المستودع {REPO_NAME} موجود، جاري تحديث الملفات...", flush=True)

    # 2. Collect all project files to commit to the repository (excluding >50MB binaries & temp files)
    files_to_commit = []
    skip_dirs = {".git", "__pycache__", "build", "dist", "models"}
    skip_files = {".gh_token", "settings.json"}

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname in skip_files or fname.endswith(".pyc") or fname.endswith(".temp_noaudio.mp4"):
                continue
            full_p = os.path.join(root, fname)
            if os.path.getsize(full_p) > 45 * 1024 * 1024:
                continue
            rel_p = os.path.relpath(full_p, base_dir).replace("\\", "/")
            files_to_commit.append((rel_p, full_p))

    # Add a .gitignore file as well
    gitignore_content = "__pycache__/\n*.pyc\n.gh_token\nsettings.json\nbuild/\ndist/\n*.temp_noaudio.mp4\n"

    # Upload files via Contents API or Git Tree API
    # Get latest commit SHA on main
    ref_r = requests.get(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/ref/heads/main",
        headers=headers,
        timeout=15,
    )
    if ref_r.status_code != 200:
        ref_r = requests.get(
            f"https://api.github.com/repos/{username}/{REPO_NAME}/git/ref/heads/master",
            headers=headers,
            timeout=15,
        )
    ref_data = ref_r.json()
    branch_ref = ref_data["ref"]  # refs/heads/main
    latest_commit_sha = ref_data["object"]["sha"]

    commit_r = requests.get(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/commits/{latest_commit_sha}",
        headers=headers,
        timeout=15,
    ).json()
    base_tree_sha = commit_r["tree"]["sha"]

    tree_items = []
    # Add .gitignore
    b_gi = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/blobs",
        headers=headers,
        json={"content": gitignore_content, "encoding": "utf-8"},
        timeout=15,
    ).json()
    tree_items.append(
        {
            "path": ".gitignore",
            "mode": "100644",
            "type": "blob",
            "sha": b_gi["sha"],
        }
    )

    for rel_p, full_p in files_to_commit:
        print(f"  ⬆️ رفع الملف: {rel_p}", flush=True)
        raw_bytes = open(full_p, "rb").read()
        b64_content = base64.b64encode(raw_bytes).decode("ascii")
        blob_r = requests.post(
            f"https://api.github.com/repos/{username}/{REPO_NAME}/git/blobs",
            headers=headers,
            json={"content": b64_content, "encoding": "base64"},
            timeout=30,
        )
        blob_r.raise_for_status()
        tree_items.append(
            {
                "path": rel_p,
                "mode": "100644",
                "type": "blob",
                "sha": blob_r.json()["sha"],
            }
        )

    new_tree = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree_items},
        timeout=25,
    ).json()

    new_commit = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/commits",
        headers=headers,
        json={
            "message": "🚀 Bluer Girl Pro v6.0 — النسخة الكاملة (تطبيق سطح المكتب + إضافة جوجل كروم + محرك 6 نماذج ذكاء اصطناعي)",
            "tree": new_tree["sha"],
            "parents": [latest_commit_sha],
        },
        timeout=20,
    ).json()

    requests.patch(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/{branch_ref}",
        headers=headers,
        json={"sha": new_commit["sha"], "force": True},
        timeout=15,
    ).raise_for_status()

    print("\n✅ تم رفع جميع الأكواد وإضافة كروم للمستودع بنجاح!", flush=True)

    # 3. Create GitHub Release v6.0.0 and attach Chrome Extension ZIP + ONNX AI Models
    print("🏷️ جاري إنشاء إصدار GitHub Release (v6.0.0) ورفع نماذج الذكاء الاصطناعي...", flush=True)
    rel_check = requests.get(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/releases/tags/v6.0.0",
        headers=headers,
        timeout=15,
    )
    if rel_check.status_code == 200:
        release_data = rel_check.json()
    else:
        rel_create = requests.post(
            f"https://api.github.com/repos/{username}/{REPO_NAME}/releases",
            headers=headers,
            json={
                "tag_name": "v6.0.0",
                "name": "🛡️ Bluer Girl Pro v6.0 — فزعة الستريمر والشاشة وقراند والأنمي",
                "body": (
                    "### 🛡️ Bluer Girl Pro v6.0 — الإصدار الكامل\n"
                    "- **تطبيق سطح المكتب الحي (Live Screen Overlay)** للألعاب وقراند والأنمي والفيديو.\n"
                    "- **إضافة جوجل كروم (Google Chrome Extension)** لتيك توك، يوتيوب، ريلز، تويتر، وتويتش/كيك.\n"
                    "- **محرك 6 نماذج ذكاء اصطناعي (GPU DirectML)** للتفريق الدقيق بين البنت والولد.\n"
                    "- **معالج ملفات الفيديو المستقل (.MP4)**."
                ),
                "draft": False,
                "prerelease": False,
            },
            timeout=20,
        )
        release_data = rel_create.json()

    upload_url_template = release_data.get("upload_url", "").split("{")[0]
    existing_assets = {a["name"] for a in release_data.get("assets", [])}

    assets_to_upload = []
    zip_ext = os.path.join(base_dir, "BluerGirl_Chrome_Extension.zip")
    if os.path.exists(zip_ext):
        assets_to_upload.append(("BluerGirl_Chrome_Extension.zip", zip_ext))

    models_dir = os.path.join(base_dir, "models")
    if os.path.exists(models_dir):
        for mfile in sorted(os.listdir(models_dir)):
            if mfile.endswith(".onnx"):
                assets_to_upload.append((mfile, os.path.join(models_dir, mfile)))

    for asset_name, asset_path in assets_to_upload:
        if asset_name in existing_assets or not upload_url_template:
            continue
        sz_mb = os.path.getsize(asset_path) / (1024 * 1024)
        print(f"  📦 رفع ملف الإصدار: {asset_name} ({sz_mb:.1f} MB)...", flush=True)
        with open(asset_path, "rb") as af:
            up_headers = {
                "Authorization": f"token {token}",
                "Content-Type": "application/octet-stream",
            }
            up_r = requests.post(
                f"{upload_url_template}?name={asset_name}",
                headers=up_headers,
                data=af,
                timeout=300,
            )
            if up_r.status_code in (200, 201):
                print(f"     ✅ تم رفع {asset_name}", flush=True)

    final_html_url = f"https://github.com/{username}/{REPO_NAME}"
    print("\n" + "=" * 65, flush=True)
    print(f"🎉 اكتمل رفع المشروع بالكامل على حسابك في GitHub!", flush=True)
    print(f"🔗 رابط المستودع: {final_html_url}", flush=True)
    print("=" * 65, flush=True)
    try:
        webbrowser.open(final_html_url)
    except Exception:
        pass


if __name__ == "__main__":
    upload_project()
