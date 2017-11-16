# Saekano-Character-Recognition

Twitter上で送られてきた画像に対して冴えカノのキャラクターかどうかを判別して、名前を当てるbotです。pythonのライブラリとしてtweepyを用いています。
冴えカノじゃなくても好きなアニメで応用可能です。

まず、入力データとしてnpyファイルを用意しますが、別のリポジトリHaircolor-Recognitionにもある` preparing_traindata.py `と同様なので省略しています。従って、入力データは予めnpyファイルとして出来ていることを前提に進めます。  

` $ python KerasCNN.py `  
によって、Keras(backendはtensorflow)でCNNを用いてクラス分類を行っています、最適化手法はSGDにしていますが今更ながらAdamの方が良かったかも。
私が行った時には、恵、詩羽、英梨々、倫也、それ以外で分類していたのでクラス数は5でした。 

` $ python Judge.py `  
で送られてきた画像がどのキャラクターなのかを判別して、その判別結果をリプライで教えてくれるものです。ただ、逐一顔だけを抽出しているので顔抽出出来ない場合の判別も含めて改良が必要です。  
CONSUMER_KEY, CONSUMER_SECRET, ACCES_TOKEN, ACCESS_SECRET, botは各自用意してください。

コメントアウトが少なくて申し訳ありませんが、不明な点は遠慮なく聞いてください。
