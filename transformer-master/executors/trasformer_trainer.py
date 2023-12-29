import os
import time

import torch
from torch import nn, device
from torch.testing._internal.common_quantization import accuracy
from torch.utils.data import DataLoader

from config.digits_dataset_cfg import cfg as dataset_cfg
from config.transformer_cfg import cfg as transformer_cfg
from config.evaluation_cfg import cfg as evaluation_cfg

from dataset.digitseq_dataset import DigitSequenceDataset
from models.transformer import Transformer
from tqdm import tqdm


class Trainer:
    def __init__(self):
        self.__prepare_data(dataset_cfg)
        self.__prepare_model()

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = DigitSequenceDataset(dataset_cfg)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=transformer_cfg.batch_size, shuffle=False,
                                           collate_fn=self.train_dataset.collate_fn)
        self.test_dataset = DigitSequenceDataset(dataset_cfg)
        self.test_dataset = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                           collate_fn=self.train_dataset.collate_fn)

    def __prepare_model(self):
        """ Подготовка нейронной сети"""
        self.model = Transformer(transformer_cfg).to(transformer_cfg.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_dataset.pad)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0, betas=(transformer_cfg.b1, transformer_cfg.b2),
                                          eps=transformer_cfg.eps_opt)

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            # реализовать сохранение модели по пути os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        """
        save_path = os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            # реализовать выгрузку весов модели по пути os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        """
        load_path = os.path.join(evaluation_cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(load_path))

    def create_masks(self, encoder_input, decoder_input):
        """
        Создает маски для трансформера, которые необходимы для корректной работы внимания (attention) в модели.

        Включает в себя следующие маски:
        1. Маска для padding входа энкодера: Эта маска используется для исключения влияния padding токенов на результаты внимания в энкодере.
        2. Маска для padding входов энкодера-декодера: Применяется в декодере для обеспечения того, чтобы padding
                                                         токены не участвовали в расчетах внимания.
        3. Маска для предотвращения утечки будущей информации в декодере: Эта маска гарантирует, что каждая позиция в
                                                декодере может взаимодействовать только с предшествующими ей позициями,
                                                что предотвращает использование "будущей" информации при генерации текущего токена.

        :param encoder_input: Тензор, представляющий последовательность на вход энкодера.
        :param decoder_input: Тензор, представляющий последовательность на вход декодера.
        :return:
            - Маска для padding входа декодера.
            - Маска для padding входов энкодера-декодера.
            - Маска для предотвращения утечки будущей информации в декодере.
        """
        encoder_padding_mask = (encoder_input != self.train_dataset.pad).unsqueeze(1).unsqueeze(
            2)  # [B, 1, 1, encoder_len]
        encoder_decoder_padding_mask = (encoder_input != self.train_dataset.pad).unsqueeze(1).unsqueeze(
            3)  # [B, 1, encoder_len, 1]

        seq_len = decoder_input.size(1)
        decoder_padding_mask = (decoder_input != self.train_dataset.pad).unsqueeze(1).unsqueeze(2)  # [B, 1, seq_len, 1]
        future_mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).bool().to(device)  # [1, seq_len, seq_len]

        return encoder_padding_mask, encoder_decoder_padding_mask, decoder_padding_mask & future_mask

    def make_step(self, batch):
        """
        Выполняет один шаг обучения для модели трансформера.

        Этапы включают:
        1. Forward Pass:
            - Получение входных данных для энкодера и декодера из батча.
            - Получение масок для обработки padding в последовательностях и для предотвращения утечки будущей информации в декодере.
            - Выполнение forward pass модели с данными энкодера и декодера.

        2. Вычисление функции потерь:
            - Функция потерь рассчитывается на основе предсказаний модели и целевых значений из батча.
            - Предсказания модели и целевые значения преобразуются в соответствующие форматы для функции потерь.

        3. Backward Pass и обновление весов:
            - Выполнение backward pass для расчета градиентов.
            - Обновление весов модели с помощью оптимизатора.

            :param batch: tuple data - encoder input, decoder input, sequence length
            :return: значение функции потерь, выход модели
            # реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        encoder_input, decoder_input, seq_len = batch
        encoder_input, decoder_input = encoder_input.to(transformer_cfg.device), decoder_input.to(transformer_cfg.device)
        target = decoder_input[:, 1:]  # Shift target sequence for teacher forcing

        # Create masks
        encoder_padding_mask, encoder_decoder_padding_mask, decoder_padding_mask = self.create_masks(encoder_input, decoder_input)

        # Forward pass
        output = self.model(encoder_input, decoder_input, encoder_padding_mask, encoder_decoder_padding_mask, decoder_padding_mask)

        # Calculate loss
        loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), target.reshape(-1))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), transformer_cfg.max_grad_norm)

        # Update weights
        self.optimizer.step()

        return loss.item(), output

    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            # реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции, accuracy, расстояние Левенштейна.
                Не считайте токены padding при подсчете точности
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        for batch in tqdm(self.train_dataloader, desc="Training", leave=False):
            loss, output = self.make_step(batch)
            total_loss += loss

            # Calculate accuracy excluding padding tokens
            predicted_classes = torch.argmax(output, dim=-1)
            target = batch[1][:, 1:]  # Exclude the start token from the target sequence
            mask = (target != self.train_dataset.pad)
            total_accuracy += accuracy(predicted_classes[mask], target[mask])

        avg_loss = total_loss / len(self.train_dataloader)
        avg_accuracy = total_accuracy / len(self.train_dataloader)

        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")

    def evaluate(self, *args, **kwargs):
        """
        Метод для оценки  модели трансформера.

        Основные шаги процесса инференса включают:
        1. Перевод модели в режим оценки (`model.eval()`), что отключает слои, работающие по-разному во время обучения и инференса (например, Dropout).
        2. Перебор данных по батчам: для каждого батча последовательно генерируются предсказания.
        3. Инференс в цикле:
            a. В качестве входа энкодера на каждом шаге используется весь input экодера, также как и на этапе обучения.
            b. В качестве входа декодера на первом шаге цикла подается одним токен - self.train_dataset.beg_seq
               Пока модель не предскажет токен конца последовательности (self.train_dataset.end_seq) или количество итераций цикла достигнет
               максимального значения transformer_cfg.max_search_len,
               на каждом шаге происходит следующее:
               - Модель получает на вход текущую последовательность и выдает предсказания для следующего токена.
               - Из этих предсказаний выбирается токен с наибольшей вероятностью (используется argmax).
               В домашнем задании с обучением перевода добавьте  softmax с температурой и вероятностное сэмплирование.
               - Этот токен добавляется к текущей последовательности декодера, и процесс повторяется.
        5. Вычисление метрик(accuracy, расстояние Левенштейна) для сгенерированной последовательности, исключая паддинг-токены из подсчета.

    # Реализуйте функцию оценки должна включать логирование значений функции потерь и точности,
          не учитывайте паддинг-токены при подсчете точности.

        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_levenshtein_distance = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.test_dataset:
                encoder_input, target_sequence, _ = batch
                encoder_input, target_sequence = encoder_input.to(transformer_cfg.device), target_sequence.to(
                    transformer_cfg.device)

                # Create masks
                encoder_padding_mask, _, _ = self.create_masks(encoder_input, target_sequence)

                # Forward pass
                output = self.model.inference(encoder_input, encoder_padding_mask)

                # Calculate loss
                loss = self.criterion(output.view(-1, output.size(-1)), target_sequence.view(-1))
                total_loss += loss.item()

                # Calculate accuracy and Levenshtein distance
                predicted_sequence = torch.argmax(output, dim=-1)
                accuracy = self.calculate_accuracy(predicted_sequence, target_sequence)
                levenshtein_distance = self.calculate_levenshtein_distance(predicted_sequence, target_sequence)

                total_accuracy += accuracy
                total_levenshtein_distance += levenshtein_distance
                total_samples += target_sequence.size(0)

        avg_loss = total_loss / len(self.test_dataset)
        avg_accuracy = total_accuracy / total_samples
        avg_levenshtein_distance = total_levenshtein_distance / total_samples

        print(f"Evaluation Loss: {avg_loss}")
        print(f"Evaluation Accuracy: {avg_accuracy}")
        print(f"Evaluation Levenshtein Distance: {avg_levenshtein_distance}")

    def fit(self, *args, **kwargs):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        best_accuracy = 0.0
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)
            self.evaluate()

            # Save the model with the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(f"best_model_{epoch}")


if __name__ == '__main__':

    trainer = Trainer()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()

