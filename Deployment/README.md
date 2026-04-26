# Huong dan cai dat va chay

Tai lieu nay huong dan tung buoc de cai dat thu vien va chay ung dung tren Windows.

## 1) Tao va kich hoat moi truong ao (venv)

Mo PowerShell tai thu muc du an (chinh la thu muc co file app.py).

```
python -m venv .venv
```

Cho phep chay script trong phien lam viec hien tai (neu can):

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Kich hoat venv:

```
& .\.venv\Scripts\Activate.ps1
```

Khi kich hoat thanh cong, dau nhac lenh se co tien to (.venv).

## 2) Cai dat thu vien

Cap nhat pip va cai dat cac thu vien trong requirements.txt:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu bi loi khi cai dat pandas hoac scikit-learn, thu cai Python 3.11 hoac 3.12.

## 3) Chay ung dung Streamlit

Chay ung dung o cong 8501:

```
python -m streamlit run app.py --server.port 8501
```

Mo trinh duyet tai dia chi:

```
http://localhost:8501
```

## 4) Loi thuong gap

- Neu PowerShell bao `python` khong ton tai, thu dung:

```
.\python.exe -m streamlit run app.py --server.port 8501
```

- Neu dang chay ma khong mo duoc trinh duyet, hay copy link trong terminal va mo thu cong.

