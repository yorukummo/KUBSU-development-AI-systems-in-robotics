import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import random
from collections import deque
from GUI import GUI
from HAL import HAL

# Параметры нейронной сети и обучения
state_size = 1  # Ошибка как единственное состояние
action_size = 3  # Действия: уменьшить, оставить, увеличить k_p
hidden_size = 64  # Количество нейронов в скрытом слое
learning_rate = 0.001  # Скорость обучения
gamma = 0.9  # Коэффициент дисконтирования
epsilon = 1.0  # Вероятность случайного действия (Exploration)
epsilon_decay = 0.995  # Уменьшение epsilon
epsilon_min = 0.1
batch_size = 32
memory_size = 10000

# Начальное значение kp
kp = 0.005

# Определяем диапазон изменения kp
kp_min = 0.001
kp_max = 0.02

# Определяем память для опыта
memory = deque(maxlen=memory_size)

# Нейронная сеть для аппроксимации Q-функции
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Инициализация модели и оптимизатора
model = QNetwork(state_size, action_size, hidden_size)
target_model = QNetwork(state_size, action_size, hidden_size)
target_model.load_state_dict(model.state_dict())  # Копируем веса
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Функция для добавления опыта в память
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Функция для выборки мини-батча из памяти
def replay():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Функция для выбора действия (epsilon-greedy)
def act(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(action_size))  # Случайное действие
    state = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        q_values = model(state)
    return torch.argmax(q_values).item()

# Основной цикл управления роботом
while True:
    frame = HAL.getImage()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    if len(contours) > 0:
        line_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(line_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        image_center = frame.shape[1] // 2
        error = cX - image_center
        max_error = frame.shape[1] // 2  # Максимальная ошибка (ширина изображения / 2)

        # Нормализуем ошибку для состояния
        state = np.array([error / max_error])

        # Выбираем действие
        action = act(state)

        # Обновляем kp в зависимости от действия
        if action == 0:
            kp = max(kp - 0.001, kp_min)
        elif action == 2:
            kp = min(kp + 0.001, kp_max)

        # Управляющий сигнал
        steering = kp * error

        # Управление роботом
        HAL.setV(1)  
        HAL.setW(-steering)

        # Вычисляем награду
        reward = -abs(error) / max_error  # Чем меньше ошибка, тем выше награда
        if abs(error) < 0.1 * max_error:
            reward += 1  # Дополнительная награда за удержание линии

        # Следующее состояние
        next_state = np.array([error / max_error])

        # Проверяем, сошел ли робот с линии
        done = False
        if abs(error) > max_error:
            reward -= 10
            done = True

        # Сохраняем опыт
        remember(state, action, reward, next_state, done)

        # Обучаем сеть
        replay()

        # Плавное уменьшение epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Копируем веса модели в target-модель каждые 100 шагов
        if len(memory) % 100 == 0:
            target_model.load_state_dict(model.state_dict())

        # Отображение
        cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

    GUI.showImage(frame)
