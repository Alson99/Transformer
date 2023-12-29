
# Трансформер
## Задача 1. Реализация Модуля SDPA (Scaled Dot-Product Attention)


<p align="center">
  <img src="images/SDPA.png" alt="Equation 4" style="vertical-align: middle;"/>
  <img src="images/AIAYN_SDPA.png" alt="Screenshot" width="300" style="vertical-align: middle;"/>
<br>
  <em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

#### Задача:
Необходимо реализовать SDPA в Pytorch, используя формулу, представленную выше. Эта задача включает в себя следующие пункты:
1.  Инициализация Softmax:
     * Реализовать инициализацию softmax в конструкторе класса SDPA (model/transformer.py).
     * Используйте `nn.Softmax` модуль с правильным измерением для применения softmax.

2. Расчет Attention Scores:
     * Написать код для расчета скалярного произведения `Q` и `K^T` (транспонированного `K`).
     * Масштабировать результат с помощью <img src="images/sqrt_dk.png"/>
     * Применить softmax к полученным значениям для получения коэффициентов внимания.
3. Расчет Выходного Тензора:
     * Реализовать умножение коэффициентов внимания на матрицу `V`.

## Задача 2. Реализация Модуля MHA 

<p align="center">
  <img src="images/MHA.png" alt="Equation 4" style="vertical-align: middle;"/>
  <img src="images/AIAYN_MHA.png" alt="Screenshot" width="300" style="vertical-align: middle;"/>
<br>
  <em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

#### Задача:
Необходимо реализовать MHA в Pytorch, используя формулу, представленную выше. Эта задача включает в себя следующие пункты:
1.  Реализация SHA (Single Head Attention):
     * Дополните инициализацию линейных преобразований для Q, K и V в конструкторе класса SHA (model/transformer.py).
     * Реализуйте метод forward, который выполняет линейные преобразования на Q, K, V и затем применяет SDPA.

2. Реализация MHA (Multi-Head Attention):
     * Инициализируйте в классе MHA список модулей SHA, равный количеству голов внимания.
     * Дополните инициализацию линейного преобразования для объединения выходов всех головок внимания.
     * Реализуйте метод forward, который вычисляет выходы каждого модуля SHA и затем объединяет их, применяя линейное преобразование.

## Задача 3. Реализация Энкодера


#### Задача:
Необходимо реализовать Энкодер в Pytorch. Эта задача включает в себя следующие пункты:
1.  Реализация Класса PositionEncoder
     * Создайте класс PositionEncoder, который генерирует матрицу positional encoding.
     * Формула Positional encoding: 
<p align="center">
<img src="images/PE.png" style="vertical-align: middle;"/>
<br>
<em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

2. Реализация FeedForward:
     * Создайте класс FeedForward, который включает в себя два линейных слоя и функцию активации ReLU.
     * Формула: 
<p align="center">
<img src="images/FFN.png" style="vertical-align: middle;"/>
<br>
<em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

3. Реализация EncoderSingleLayer:
     * Создайте класс EncoderSingleLayer, содержащий один слой энкодера. Он должен включать Multi-Head Attention (MHA), два слоя Layer Normalization и один Feed Forward слой.
     * В процессе обработки должны использоваться residual connections и последующая нормализация. 
<p align="center">
<img src="images/EncoderSingleLayer.png" style="vertical-align: middle;"/>
<br>
<em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>

3. Реализация Класса Encoder
     * Создайте класс Encoder, который состоит из нескольких EncoderSingleLayer.
     * Он должен последовательно применять эти слои к входным данным.
<p align="center">
<img src="images/Encoder.png" style="vertical-align: middle;"/>
<br>
<em> "Attention Is All You Need" by Ashish Vaswani et al., published in 2017</em>
</p>


#### Подсказки и Рекомендации

     * Обратите внимание на размерности входных и выходных данных каждого слоя.
     * Для линейных слоев в FeedForward используйте увеличение размерности в 4 раза между первым и вторым слоем.


## Задание 4. Реализация Трансформера

<p align="center">
  <img src="images/AIAYN_decoder.png" alt="Decoder" style="vertical-align: middle;"/>
</p>
<p align="center">
  <img src="images/AIAYN_transformer.png" alt="Decoder" style="vertical-align: middle;"/>
</p>

#### Задача:
Реализовать трансформер и обучить его на искуссвтенно сгенерированных данных DigitSequenceDataset. После обучения трасформер должен повторять последовательность цифр, которую ему подали на вход энкодера.

1.  Добавление Масок в SDPA
   * добавьте маску для предотвращения утечки будущей информации в декодере
   <p align="center">
      <img src="images/mask.png" alt="mask" style="vertical-align: middle;"/>
   </p>
    
   * добавьте маску для padding токенов с декодера и энкодера-декодера
   <p align="center">
      <img src="images/padding_token.png" alt="mask" style="vertical-align: middle;"/>
   </p>

2.  Добавьте Dropout
    * После сложения эмбеддинга слова и PE вектора
    * После вычисления SDPA
    * После вычисления MHA, перед Add&Norm слоем
    * После вычисления FeedForward, перед Add&Norm слоем
    
3.  Реализация DecoderSingleLayer
    * Инициализация Multi-Head Attention для self-attention.
    * Инициализация LayerNorm и Dropout после self-attention.
    * Инициализация Multi-Head Attention для внимания между энкодером и декодером.
    * Инициализация LayerNorm и Dropout после внимания между энкодером и декодером.
    * Инициализация Feed Forward слоя.
    * Инициализация LayerNorm и Dropout после Feed Forward слоя.
    * Реализация forward pass, сочетающего все эти слои.

4.  Реализация Decoder
    * Инициализация нескольких слоев DecoderSingleLayer.
    * Реализация forward pass через все слои декодера.
    
5.  Класс Transformer
    * Инициализация слоя эмбеддинга и position encoding.
    * Инициализация энкодера.
    * Инициализация декодера.
    * Инициализация линейного слоя для выхода.
    * Реализация forward pass через энкодер и декодер, включая применение эмбеддингов и position encoding.

6.  Реализуйте класс Trainer для обучения транформера и обучите трансформер на искусственно 
    * Реализация функции для генерации масок для предотвращения утечки будущей информации в декодере, для padding токенов с декодера и энкодера-декодера
    * Реализуйте функции для обучения трансформера
    * Реализуйте функцию инференса трасформера
    * Для получения предсказаний используйте argmax
    * В качестве метрики возьмите accuracy
    * Реализуйте логирование с помощью Neptune

#### Подсказки и Рекомендации
    * Обратите внимание, что для данной задачи нужно использовать одинаковые матрицы эмбеддингов для энкодера, декодера и выходного слоя.

## Задание 5. Обучение модели Трансформера на перевод

#### Задача:
Обучить трасформер на задачу перевода с английского на русский

1.  Возьмите реализацию датасета TranslationDataset. Данный датасет будет выдавать вам предложения на английском и русском языке
2.  Обучите трасформер на перевод с английского на русский
3.  Реализуйте softmax с температурой
4.  Напишите получение предсказаний с использованием random sampling
5.  Реализуйте метод metrics/levenshtein_distance. Добавьте подсчет этой метрики при обучении и валидации  

