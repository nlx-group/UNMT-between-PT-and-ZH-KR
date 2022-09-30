# UNMT-between-PT-and-ZH-KR

This dissertation presents a comparative and reproduction study on Unsupervised Neural Machine Translation techniques in the pair of languages Portuguese (PT) $\to$ Chinese (ZH) and Portuguese (PT) $\to$ Korean(KR).

We chose these language-pairs for two main reasons. The first one refers to the importance that Asian languages play in the global panorama and the influence that Portuguese has in the southern hemisphere. The second reason is purely academic. Since there is a lack of studies in the area of Natural Language Processing (NLP) regarding non-Germanic languages, we focused on studying the influence of non-supervised techniques in under-studied languages. 

In this dissertation, we worked on two approaches: (i) Unsupervised Neural Machine Translation; (ii) the Pivot approach. The first approach uses only monolingual corpora. As for the second, it uses parallel corpora between the pivot and the non-pivot languages.

The unsupervised approach was devised to mitigate the problem of low-resource languages where training traditional Neural Machine Translations was unfeasible due to requiring large amounts of data to achieve promising results. As such, the unsupervised machine translation only requires monolingual corpora. 
In this dissertation we chose the implementation of \citet{DBLP:journals/corr/abs-1710-11041} to develop our work.

Another alternative to the lack of parallel corpora is the pivot approach. In this approach, the system uses a third language (called pivot) that connects the source language to the target language. The reasoning behind this is to take advantage of the performance of the neural networks when being fed with large amounts of data, making it enough to counterbalance the error propagation which is introduced when adding a third language.

The results were evaluated using the BLEU metric and showed that for both language pairs Portuguese $\to$ Chinese and Portuguese $\to$ Korean, the pivot approach had a better performance making it a more suitable choice for these dissimilar low resource language pairs.
