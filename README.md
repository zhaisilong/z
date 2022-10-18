# Science Tools

[![PyPI version](https://badge.fury.io/py/sci-ztools.svg)](https://badge.fury.io/py/sci-ztools)

Zhai Silong's Greate tools for science research.

## Usage

### utils

Core

### pep

Pep tools

### chem

chem tools

## 上传

- [先注册一个 PyPi 的测试帐号](https://test.pypi.org/account/register/)
- 获取帐号的 [api token](https://test.pypi.org/manage/account/token/)
  - 写入 `$HOME/.pypirc`

```bash
# 安装或更新 setuotools 和 wheel
python3 -m pip install  --upgrade setuptools wheel
# cd 到 setup.py 所在目录
python3 setup.py sdist bdist_wheel  # 产生一个 dist 文件夹，用于本地安装
# 更新 twine
python3 -m pip install --user --upgrade twine
# 使用 twine 上传
python3 -m twine upload --repository testpypi dist/*
# 正式上传
python3 -m twine upload dist/*
# 测试
pip install example-pkg-YOUR-USERNAME
```

## Reference

- [python--如何将自己的包上传到PyPi并可通过pip安装](https://blog.csdn.net/yifengchaoran/article/details/113447773)