# Tanka

- Wengertリスト（tapeとも呼ばれる）は、行った計算をリストに追加していくだけで、任意の計算グラフに対して逆伝搬を正しく行うことができる
  - ↓はリンク付きノード
  - 「関数が変数(input)をもつ、変数が関数(creator)を持つ」ことによって、端末から起点までの計算グラフを逆向きに辿れるようになる
  - ![image-20201017213919883](/Users/reon.nishimura/Library/Application Support/typora-user-images/image-20201017213919883.png)



### メモリ管理と循環参照

- CPythonのメモリ管理は「参照カウント」と「GC」の２つがある

  - 基本、参照カウント

- 参照カウント

  - 参照カウントが０の状態で生成される
  - 他のオブジェクトから参照されるとカウントが１上がる
  - 参照がなくなると１減る
  - ０になると、インタプリタから消去される
  - オブジェクトは不要になると即座にメモリから消去される
  - `sys.getrefcount(x) - 1`で参照カウントがわかる
    - `-1`してる理由は、getrefcountに渡したときにカウントが１つ増えるから

- 循環参照

  - ![image-20201028005951916](/Users/reon.nishimura/Library/Application Support/typora-user-images/image-20201028005951916.png)
  - →はa = b = c = Noneとしたとき、循環参照しているのでカウントが残ってしまう
  - `gc.collect()`を行うことで明示的に削除できる
  - 基本的に循環参照を無くす
  - 関数と変数に循環参照が生じてる
    - ![image-20201028010127878](/Users/reon.nishimura/Library/Application Support/typora-user-images/image-20201028010127878.png)

- 弱参照

  - ```python
    b = weakref.ref(a)

    print(b) # ポインタ
    print(weakref.getweakrefcount(a))  # 1
    print(b()) # 実体
    a = None
    print(weakref.getweakrefcount(a))  # 0
    print(b) # <weakref at 0x13338c720; dead>

    ```


### パッケージとしてまとめる

- モジュール：Pythonのファイル
- パッケージ：複数のモジュールをまとめたもの
- ライブラリ：複数のパッケージをまとめたもの
