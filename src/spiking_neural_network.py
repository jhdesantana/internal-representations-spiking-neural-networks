from typing import Any

import numpy as np
import snntorch.functional as SF
import torch
import torch.nn as nn
from snntorch._neurons import SpikingNeuron
from snntorch.functional.loss import LossFunctions
from torch._tensor import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Snn(nn.Module):
    def __init__(self, spk_neuron: SpikingNeuron, num_units_layers: list):
        super().__init__()
        self.num_units_layers = num_units_layers
        self.num_layers = len(self.num_units_layers) - 1
        self.neuron = spk_neuron
        self.layers = nn.ModuleList()
        self.neurons_layers = nn.ModuleList()

        for k in range(self.num_layers):
            self.layers.append(
                nn.Linear(self.num_units_layers[k], self.num_units_layers[k + 1])
            )
            self.neurons_layers.append(self.neuron)

    def forward(self, input: Tensor, **kwargs: dict[str, Any]) -> list[Tensor]:
        num_steps = kwargs.get("num_steps", 25)
        mem = kwargs.get("mem", [0] * self.num_layers)

        batch_size = input.size(0)
        layers = [
            torch.zeros(
                (self.num_layers, num_steps, batch_size, self.num_units_layers[k])
            )
            for k in range(1, self.num_layers + 1)
        ]

        for step in range(num_steps):
            input_layer = input
            for k in range(self.num_layers):
                cur = self.layers[k](input_layer)
                spk, mem[k] = self.neurons_layers[k](cur, mem[k])

                layers[k][0, step] = spk
                layers[k][1, step] = mem[k]
                layers[k][2, step] = cur

                input_layer = spk

        return layers

    def train(
        self,
        loss: LossFunctions,
        optimizer: Optimizer,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        batch_size: int,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        training_time = len(train_loader)
        train_loss = np.zeros((epochs, training_time))
        train_acc = np.zeros((epochs, training_time))
        test_acc = np.zeros((epochs, training_time))
        test_loss = np.zeros((epochs, training_time))

        w_0 = np.zeros(
            (
                epochs,
                training_time,
                torch.flatten(self.state_dict()["layers.0.weight"]).shape[0],
            )
        )

        w_1 = np.zeros(
            (
                epochs,
                training_time,
                torch.flatten(self.state_dict()["layers.1.weight"]).shape[0],
            )
        )

        w_2 = np.zeros(
            (
                epochs,
                training_time,
                torch.flatten(self.state_dict()["layers.2.weight"]).shape[0],
            )
        )

        for e in range(epochs):
            for b_idx, (train_inputs, train_labels) in enumerate(train_loader):
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)

                layers = self.forward(train_inputs.view(batch_size, -1))

                train_outputs = layers[-1][0]
                train_loss_batch = loss(train_outputs, train_labels)

                optimizer.zero_grad()
                train_loss_batch.backward()
                optimizer.step()

                train_loss[e, b_idx] = train_loss_batch.item()

                with torch.no_grad():
                    test_inputs, test_labels = next(iter(test_loader))
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device)

                    layers = self.forward(test_inputs.view(batch_size, -1))

                    test_outputs = layers[-1][0]
                    test_loss_batch = loss(test_outputs, test_labels)
                    test_loss[e, b_idx] = test_loss_batch.item()

                train_acc[e, b_idx] = SF.accuracy_rate(train_outputs, train_labels)

                test_acc[e, b_idx] = SF.accuracy_rate(test_outputs, test_labels)

                w_0[e, b_idx] = torch.flatten(
                    self.state_dict()["layers.0.weight"]
                ).numpy()

                w_1[e, b_idx] = torch.flatten(
                    self.state_dict()["layers.1.weight"]
                ).numpy()

                w_2[e, b_idx] = torch.flatten(
                    self.state_dict()["layers.2.weight"]
                ).numpy()

                if b_idx % 20 == 0:
                    print("Epoch:", e + 1, "batch_idx:", b_idx)

        train_loss = np.concatenate(train_loss, axis=0)
        train_acc = np.concatenate(train_acc, axis=0)
        test_loss = np.concatenate(test_loss, axis=0)
        test_acc = np.concatenate(test_acc, axis=0)

        return train_loss, train_acc, test_loss, test_acc, w_0, w_1, w_2

    def evaluation(
        self,
        data_loader: DataLoader,
        batch_size: int,
        device: str,
        **kwargs: dict[str, Any]
    ) -> float:
        state_dict = kwargs.get("checkpoint", self.state_dict())
        self.load_state_dict(state_dict)

        with torch.no_grad():
            total = 0
            acc = 0
            for _, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                layers = self.forward(inputs.view(batch_size, -1))

                spk3 = layers[-1][0]
                acc += SF.accuracy_rate(spk3, labels) * spk3.size(1)
                total += spk3.size(1)

        return acc / total

    def predictions(
        self, data_loader: DataLoader, batch_size: int, device: str
    ) -> tuple[np.ndarray, np.ndarray]:
        num_batchs = len(data_loader)
        pred_label = np.zeros((num_batchs, batch_size))
        true_label = np.zeros((num_batchs, batch_size))

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                layers = self.forward(inputs.view(batch_size, -1))
                outputs = layers[-1][0]

                # _: number of spikes of (max_idx)th neuron in the output layer
                # max_idx: prediction
                _, max_idx = outputs.sum(dim=0).max(1)

                max_idx = max_idx.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                pred_label[batch_idx, :] = max_idx
                true_label[batch_idx, :] = labels

        return np.hstack(pred_label), np.hstack(true_label)

    def movie(
        self,
        checkpoint: dict,
        frames: list[Tensor],
        device: str,
        **kwargs: dict[str, Any]
    ) -> list[list[np.ndarray]]:
        state_dict = checkpoint["model_state_dict"]
        self.load_state_dict(state_dict)

        num_steps = kwargs.get("num_steps", 1)

        layers_rec = []
        mem = [0] * self.num_layers
        num_frames = len(frames)
        for n in range(num_frames):
            frame = frames[n]
            frame = DataLoader(frame, batch_size=1, shuffle=False, drop_last=False)
            frame = next(iter(frame))
            frame = frame.to(device)

            layers = self.forward(
                frame.view(frame.size(0), -1), num_steps=num_steps, mem=mem
            )

            # Recurrent dynamics.
            mem = [layers[k][1, -1, 0, :] for k in range(self.num_layers)]

            layers_rec.append(
                [
                    layers[k].squeeze().cpu().detach().numpy()
                    for k in range(self.num_layers)
                ]
            )

        return layers_rec
