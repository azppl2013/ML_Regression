Regression : House Sale Price Prediction Challenge 實驗記錄
===

老師好、大家好，我是電子三丙的同學（108360721 陳靖元），<br>
會跨班來選這門課是因為我本身對於人工智慧相當感興趣，未來也想繼續往計算機科學這條路走，<br>
但因為沒有相關的經驗，想透過這門機器學習課程，來累積紮實的基礎。<br>
<br>
以下是我這幾個禮拜的研究與學習過程：
<br><br>

首先，建立 Python 開發環境
--
我是一位Python新手，在修這門課以前並沒有學過Python程式語言，因此，在這第一步就遭遇難題，<br>
原本以為就像 Visual Studio、XCode......等方便的軟體，有現成的安裝包端上桌，<br>
但翻閱了網路上的 TensorFlow、PyTorch 環境建置教學文，發現事情並沒有我想的那麼簡單，<br>
<br>
需要先下載 Anaconda ，這個軟體可謂是萬物的起源、一切的基底，<br>

![pixnet](https://i.ytimg.com/vi/Q0jGAZAdZqM/maxresdefault.jpg) <br>

下載連結：https://www.anaconda.com/products/individual <br>

Anaconda 安裝教學：https://cvfiasd.pixnet.net/blog/post/175016013-anaconda%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8 <br>

<br>
在安裝時需要特別注意自己安裝的 Python 版本，我在這邊打轉了很久， <br>
因為 Python 與 TensorFlow 和等等會遇到的 Cuda ......等，對於版本號的相對應是非常重要的，<br>
只要安裝的版本一不匹配，安裝程式便會無情的給出錯誤碼。<br>
<br>
我安裝的版本分別是：<br>

1. Python 3.8
2. Cuda 11.5
3. Cudnn 8.3
4. TensorFlow-gpu 2.7
<br>

在安裝好 Anaconda 之後，接下來需要使用 Anaconda Prompt 安裝我們需要的各種函式庫，<br>
或者使用介面較為人性化的 Anaconda Navigator 安裝也可以，但是在這個平台上（Conda）安裝的套件往往都不是最新的，<br>
而以上 Anaconda Prompt 與 Anaconda Navigator 都在安裝 Anaconda 時便會一起安裝好。 <br>
<br>
這邊以 Anaconda Prompt 安裝舉例，<br>

 ![AP](https://user-images.githubusercontent.com/95005809/143386140-87bffa03-f2c8-4500-881b-c888e917a7aa.png) <br>
 
我使用到的套件包括： <br>
1. jupeter notebook
2. Numpy
3. Matplotlib
4. Pandas
5. Seaborn
6. Sklearn
7. Tensorflow-gpu
8. Keras

<br>


在安裝多個函式庫以前，需要先建置一個虛擬環境。<br>
透過搜索網路上的資料，大多人會根據不同的開發需求，建立對應的虛擬環境， <br>
這個部分我是使用 Anaconda nevigator 的圖形化介面來建立。<br>
![nav-defaults](https://user-images.githubusercontent.com/95005809/143391648-9934a777-8406-4ff1-996b-4363443957c8.png)

<br>

再次提醒，版本對應的問題很重要，記得先查詢好對應的環境版本號，不然容易走了彎路。<br><br>
在虛擬環境上安裝函式庫的部分，先打開 Anaconda Prompt ，會看到如上圖類似命令提示字元（cmd）的介面，<br>
在 Anaconda Prompt 中，主流安裝函式庫的方式是使用 pip 來進行，<br>
pip 是 Python 中的標準庫管理器。它允許你安裝和管理不屬於 Python 內建函示庫的其它插件。 <br><br>
要在 Anaconda Prompt 安裝插件包需使用指令：

> pip install (你要安裝的插件包名字)

<br>


<br>
把上面提到的所有套件都使用這條指令安裝好後，還沒有結束！<br>
注意到我們安裝的第七個套件「 Tensorflow-gpu 」，<br>
名稱後面多了一個「 gpu 」，代表這個版本的 Tensorflow 是支援使用 gpu 來運算，<br>
在訓練的過程中，可以透過 cuda 進行大幅的加速，<br>
經過我的實測，速度差距甚至高達數十～數百倍之譜。（以此專案為例）<br><br>

剛剛所提到的 CUDA 運算單元，需要相對應版本的開發者元件，才有辦法使 Tensorflow-gpu 調用，<br>
否則 Tensorflow 就只能使用 CPU 運算。<br>
總而言之，在安裝好前段的八個函式庫後，還需要再安裝「 CUDA 」與「 CUDNN 」。<br>
此外 CUDA 與 CUDNN 也需要配合對應的 Tensorflow 與 Python 版本來安裝。<br><br>
Tensorflow-gpu 的部分我是參考這篇文章安裝：<br>
https://cvfiasd.pixnet.net/blog/post/175023846-windows%E5%AE%89%E8%A3%9Dtensorflow%E6%95%99%E5%AD%B8 <br>
<br>
做完以上步驟之後，恭喜你，學習機器學習的大門已為你敞開。<br><br>



房價回歸（一）—— 整理訓練資料
--

<br>

在我們獲取一大堆資料時，把它們全部丟到程式裡，讓它自己尋找規律是不可行的，<br>
需要先進行資料的預處理。<br>
在這個步驟中，我們要判別什麼樣的數據對於結果會造成關鍵性的影響、以及什麼樣的數據是來亂的，<br>
這個部分我是參考 Kaggle 上一位高手的文章：<br>
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python <br>
<br>
接下來我把過程以中文簡述一遍：<br>

在整個處理資料的過程中，我們需要進行幾個必要的動作後才能將資料拿去訓練，<br>
1. 瞭解問題
2. 觀察單一變數
3. 多變數的比較
4. 清理資料
5. 驗證假設
<br><br>
第一個部分，我們知道房價預測要求的結果是一列數字，這串數字是根據 "test.csv" 檔給我們的房屋數據，推算出每一間房的價格，<br>
那麼在 "train.csv" 中，訓練的資料已經給我們許多房屋條件、以及它的價格，我們將以這份 csv 檔來獲取訓練資料。<br>
<br>
最一開始，我們需要先了解，手邊有的數據具有什麼樣的元素，<br>

![擷取](https://user-images.githubusercontent.com/95005809/143772950-b40a8cd9-4703-4f0c-933f-1adefa0b8bb2.PNG) <br>
<br>

根據老師給予的對照表，可以知道每一個代號所代表的參數意義，<br>
接著在第二步驟與第三步驟中，我們要找出參數間的關聯性，<br>
以提取出與 " price " 最有關連的參數。<br>

<br>

![螢幕擷取畫面 2021-11-28 225435](https://user-images.githubusercontent.com/95005809/143773135-a68f3c50-c90f-4be4-a003-b1c2c6b70731.jpg)

上圖是多變數分析的重點之一，使用了相關性矩陣尋找變數之間的相關性，<br>
顏色越淡，代表相關程度越高、數字低於 0 則為負相關 <br>
這個部分，在學習了解原理時，我參考了這篇文章：<br>
https://www.sciencedirect.com/topics/mathematics/correlation-matrix <br>
<br>
![螢幕擷取畫面 2021-11-28 230029](https://user-images.githubusercontent.com/95005809/143773355-1d981c43-414f-468b-b96c-5e875f55cf96.jpg)
<br>
我後來使用了上圖的程式，過濾出了相關係數大於 0.5 的參數，作為我們主要處裡的對象。<br><br>


搞定離群值與資料標準化！
--
<br>
承上段，最後選定了以下參數來觀察：<br>
'price', 'sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'sqft_basement'<br>
<br>
分別把各個參數進行 Scatter plots 圖的相互比較，獲得這張壯觀的圖 <br>

![螢幕擷取畫面 2021-11-28 232641](https://user-images.githubusercontent.com/95005809/143774416-91d9ae13-e515-4157-a26f-17a09fab7d0e.jpg)


 <br>

可以容易的看出變數間的相互關係為何，<br>
<br>

<br>

![螢幕擷取畫面 2021-11-28 232836](https://user-images.githubusercontent.com/95005809/143774509-12f745cc-4bfd-4f85-9343-c6467d8af33f.jpg)
<br>
![螢幕擷取畫面 2021-11-28 232906](https://user-images.githubusercontent.com/95005809/143774532-ffed6afb-fd42-4c50-ad38-4fc92b2df121.jpg)
<br>
![螢幕擷取畫面 2021-11-28 232938](https://user-images.githubusercontent.com/95005809/143774551-46a3840d-3362-45cd-9fe4-2376cc184ef2.jpg)
<br>
![螢幕擷取畫面 2021-11-28 233041](https://user-images.githubusercontent.com/95005809/143774600-89a46369-a549-4480-832f-6127f51811bb.jpg)





<br><br>

開始訓練模型！
--
![image](https://user-images.githubusercontent.com/95005809/143774619-6c533129-854f-4867-9c61-8f6ed4d26c10.png) <br>

這個部分我使用了 relu 曲線來近似數據的形狀，並遞迴了七次 <br>

![螢幕擷取畫面 2021-11-28 233315](https://user-images.githubusercontent.com/95005809/143774703-6acad785-9480-4076-9a84-511d8a71747e.jpg)

<br>

由於時間倉促，所以只做到了這裡，<br>
在 kaggle 得到的 loss 有 156397.34 。




<br>

心得
--

實作時，因為是完全的 Python 新手，所以遭遇了很多困難、與期中考撞期<br>
做到這個步驟時，時間就已經所剩無幾。<br>
<br>
同時上了李宏毅教授與廖元甫教授的課程，對於理論層面確實有很大的幫助，<br>
在實作時也更好理解自己在做什麼、需要做什麼，<br>
在下次的 Kaggle 競賽，相信我的表現一定能比這次更好，<br>
我也將會下更多功夫來學習這門課，精進自身的機器學習理論與實作能力。<br>
<br><br>







