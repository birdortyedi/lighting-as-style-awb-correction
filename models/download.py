import requests


def download_from(url, filepath):
    r = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


if __name__ == '__main__':
    model_64_dst_url = "https://www.dropbox.com/s/2sj43ym2sr96p3o/ifrnet_p_64_D_S_T.pth?dl=1"
    model_128_dst_url = "https://www.dropbox.com/s/u1vt9uz0h1412kg/ifrnet_p_128_D_S_T.pth?dl=1"
    download_from(model_64_dst_url, "models/ifrnet_p_64_D_S_T.pth")
    download_from(model_128_dst_url, "models/ifrnet_p_128_D_S_T.pth")
