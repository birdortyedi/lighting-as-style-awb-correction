import requests


def download_from(url, filepath):
    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    r = requests.get(url, stream=True, headers=headers)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


if __name__ == '__main__':
    model_64_dst_url = "https://www.dropbox.com/s/sk7pybq3xrnithc/ifrnet_p_64_D_S_T.pth?dl=0"
    model_128_dst_url = "https://www.dropbox.com/s/f1tthna0hri59l6/ifrnet_p_128_D_S_T.pth?dl=0"
    download_from(model_64_dst_url, "models/ifrnet_p_64_D_S_T.pth")
    download_from(model_128_dst_url, "models/ifrnet_p_128_D_S_T.pth")
