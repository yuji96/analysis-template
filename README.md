仮想環境のカーネルを jupyter に登録するコマンド

```
activate
uv run ipython kernel install --user --name hoge
jupyter kernelspec list
```

消すとき
```
uv run jupyter kernelspec remove hoge
```
