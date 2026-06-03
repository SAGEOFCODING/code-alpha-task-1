import urllib.request
import os

screens = [
    {
        "name": "Iris Classification Dashboard",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLvbLiBGJSqP1Vaz_byJebAMz7YngSMIwheKwEKxG5TEROn90ZVY2okZKVefpb6SFrBmFotrESFMdBIH8_6mGKkHpJ6eGmss6HS2O__Q8LGIDSq21XKqUvGfYr5yifGMnh03HCbWxxTyVpFNKtm5kE8ub54ciDpBNatyaQfz0IoYD_tzFiICLUq7DLTZDjh5D9At0gbomtfQgUsy8bD_Tp4DetcMBKj3pb7w4s4rdUxsE3bCxCO52tVhccI",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sX2VlNDMyMmJhYTRlNzRkZjc5Nzc0M2MyNWEwOTcwYmFkEgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=96797242"
    },
    {
        "name": "Iris Classification Dashboard - Dark Mode",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLuC_IcYObymHVLl6CcIkv4LE4_lxAO9IEiYkqwD07MMDN0MAI2Z3B_9uyeQBC7t-5Ox0_RhK2lNqTaSNT6xdwIOGwIes7IkMNfEHiSsK075r_sB5iLDXY9hrSacYZ9QfrpJRW97LVPAu1fQjuVpOPcqrN1liPn0A2qxZzMeDWFFZXiQStLlI4LtU77iXA_FYs30lPMkWY9nz2RifN19HdHgY5B4hNvark5Uxz27_2vs-6hMNqCX7utWnBQ",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzAwMDY1MzU2N2QxNGU4YTQwMmE5YmQ3MWFiMzIyNTA4EgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=89354086"
    },
    {
        "name": "Iris Dashboard - Bauhaus Style",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLtSFLGO4-aZeEZWd8SjCRbpyYNQKDhR6ozjKiJPJUTnWJPtD17luR2LSyytEpyIWAB9IA9HJtzFIPCWNmkcnc5ezKMN_sX6jbLczAVVLPx_rHrRUy7jXVH2eLe04x6McyZ-5nDpGAQVoDVb3a4mh6szLTZapBFYt7UrgNGyrv9CQLA4lqPjrXRFIhBVgLfFohUVVPQV7r9yqh4SuZuff3VO5a5IMSKWKONsKpXs-RZ-E34VAkV5GBMyOEo",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzc3NjFkZTRmNDhjYzQ4Yzg5OTg2NzMyMGFlNTRlMTMxEgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=89354086"
    },
    {
        "name": "Iris Professional Dashboard",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLvlQkZwYfIbHDO_57LHd1-_rcvC0IbFXK0g9IGAVqdsGgvMrZq-RnSb1JITh2sW_TPYR7irJWvUgkpcmdYtzd25h6yEA5M-U_cpUEU9DlEzUWk34MSbOQxzQaCMAzsLoY_ZfD5xFPm3KCBbasMvFsqJy-baZbrZmW6opNGSJyXVaY5YqVVLH097dGdMoqIKs83FhbPUU8YhM9POsx5LVImCOFNqkpayWqrc7tlH5QwL5pAL0a5-C5-Ekl4",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzRiMWJlYWZmMjM3YTRmYjU4ZTEzOGI0MzAzODFkMDI2EgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=89354086"
    },
    {
        "name": "Iris Advanced Dashboard - Light Mode",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLv8L5jzOzkPWZfOj11CzjlP4c0lMq77zX0m4ZDAEjkxLmyhY7a2m0oIOivru7qS6CYRPAzRXXxmEU6n8slkTq5foBpWX0Nx12Y88Ho2es4lCyjyVGUujyVupBdu3OyXku1UnzBq0WKI_mF3xzqiw6YuEJ785W6SsvZGrM6M4dizNZVc2eDU5DQZH6eTd-7CRO27zSYt4xAZwBBUpPcCMo6IXEc6vkvlIl2VlAOP1sH7VoyHTXuENMBHEf0",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzMyMWU4OWIyMjU4MTRjZWZiNWI2ZTRmOGE1MmJkOTEwEgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=89354086"
    },
    {
        "name": "Iris Dashboard - Animated Bauhaus Style",
        "image_url": "https://lh3.googleusercontent.com/aida/AP1WRLu0VQfAdIWAgQofszWqp42HNogYuGgMLVhWa2ed6zNrh1UDixXQJRWEjnJZ4WYPbiXqW3v9uT9vFCNUstlalEBHbv0DjRgNTP5NUeAn-z49aIqMGtaQ6e_6JudaG-nU9Al2TiC3nFR6dfTTEdazyv9dk7Jl48XpGz3YJiYKKIBB-hCrKAzjkOdbD53UQ_lxj_JAwK0iRjTYOnKf2flBoynJAca61YibVTbmHqGumjpQkMG4s5jws1jxL5Y",
        "html_url": "https://contribution.usercontent.google.com/download?c=CgthaWRhX2NvZGVmeBJ7Eh1hcHBfY29tcGFuaW9uX2dlbmVyYXRlZF9maWxlcxpaCiVodG1sXzVkOTk4NDVjNzg0OTRjMTNhM2IyMzA1OWQ5OWY1ZGMwEgsSBxDMuqza5AIYAZIBIwoKcHJvamVjdF9pZBIVQhM3MjU3MzE4NzA2NTk3NzMyNjU5&filename=&opi=89354086"
    }
]

out_dir = "stitch_exports"

for s in screens:
    name_clean = s['name'].replace(" ", "_").replace("-", "").lower()
    
    img_path = os.path.join(out_dir, f"{name_clean}.png")
    html_path = os.path.join(out_dir, f"{name_clean}.html")
    
    print(f"Downloading {s['name']}...")
    urllib.request.urlretrieve(s['image_url'], img_path)
    urllib.request.urlretrieve(s['html_url'], html_path)

print("Done downloading all screens!")
