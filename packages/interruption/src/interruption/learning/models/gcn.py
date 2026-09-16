"""
A graph convolution neural network model
to estimate the expected cost of a state.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    TransformerConv
)
from interruption.learning.utils import prepare_gcn_input, convert_batch_format
from railroad.environment.procthor.scenegraph import SceneGraph


class AnticipateGCN(nn.Module):
    name = "AnticipateGCN"

    def __init__(self, args=None):
        super(AnticipateGCN, self).__init__()
        torch.manual_seed(8616)
        self._args = args

        # self.fc1 = nn.Linear(775, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 64)
        self.conv1 = TransformerConv(775, 256, edge_dim=1)
        self.conv2 = TransformerConv(256, 128, edge_dim=1)
        self.conv3 = TransformerConv(128, 32, edge_dim=1)
        self.conv4 = TransformerConv(32, 8, edge_dim=1)
        self.fc = nn.Linear(8*2, 1)

        # self.fc1bn = nn.BatchNorm1d(512)
        # self.fc2bn = nn.BatchNorm1d(256)
        # self.fc3bn = nn.BatchNorm1d(128)
        # self.fc4bn = nn.BatchNorm1d(64)
        self.conv1bn = nn.BatchNorm1d(256)
        self.conv2bn = nn.BatchNorm1d(128)
        self.conv3bn = nn.BatchNorm1d(32)
        self.conv4bn = nn.BatchNorm1d(8)

    def forward(self, data, device):
        h = data['latent_features'].type(torch.float).to(device)
        edge_data = data['edge_data']
        edge_features = data['edge_features'].type(torch.float).to(device).unsqueeze(1)
        edge_index = edge_data.type(torch.long).to(device)
        batch_index = data['batch_index'].to(device)

        # h = F.leaky_relu(self.fc1bn(self.fc1(h)), 0.1)
        # h = F.leaky_relu(self.fc2bn(self.fc2(h)), 0.1)
        # h = F.leaky_relu(self.fc3bn(self.fc3(h)), 0.1)
        # h = F.leaky_relu(self.fc4bn(self.fc4(h)), 0.1)

        # Convolution Layers
        h = F.leaky_relu(self.conv1bn(self.conv1(h, edge_index, edge_features)
                                      ), 0.1)
        h = F.leaky_relu(self.conv2bn(self.conv2(h, edge_index, edge_features)
                                      ), 0.1)
        h = F.leaky_relu(self.conv3bn(self.conv3(h, edge_index, edge_features)
                                      ), 0.1)
        h = F.leaky_relu(self.conv4bn(self.conv4(h, edge_index, edge_features)
                                      ), 0.1)

        # Pooling
        h = torch.cat(
            [global_mean_pool(h, batch_index),
             global_add_pool(h, batch_index)],
            dim=1
        )
        ec = self.fc(h)
        return ec

    def loss(self, nn_out, data, device="cpu", writer=None, index=None):
        y = data.y.to(device)
        op = nn_out[:, 0]
        # mean absolute error
        loss = nn.L1Loss()
        loss_tot = loss(op, y)
        # Logging
        if writer is not None:
            writer.add_scalar("Loss/total_loss", loss_tot.item(), index)

        return loss_tot

    # @learning.logging.tensorboard_plot_decorator
    # def plot_images(self, fig, image, out, data):
    #     pred_cost = (out).cpu().numpy()
    #     true_cost = data.y
    #     axs = fig.subplots(1, 1)
    #     axs.imshow(image)
    #     axs.set_title(f"true cost: {true_cost} | predicted cost: {pred_cost}")

    @classmethod
    def get_net_eval_fn(cls, network_file: Path | str, device: torch.device) -> "GCNEvalFn":
        """
        Returns a learned function that maps a SceneGraph representation of
        the environment state to the expected value over the task distribution.
        Callable directly on a single SceneGraph; its `.batch` method evaluates a
        list of SceneGraphs with one model call instead of one call per graph.
        """
        # load trained gcn
        model = AnticipateGCN()
        model.load_state_dict(torch.load(network_file, map_location="cpu"))
        model.to(device)
        model.eval()
        return GCNEvalFn(model, device)


class GCNEvalFn:
    """
    Callable wrapper around a loaded AnticipateGCN: evaluates one SceneGraph via
    __call__, or a list of them in a single batched model call via .batch. Only
    a plain function is otherwise interchangeable with the single-graph form
    (Callable[[SceneGraph], float]), so this exists mainly to carry .batch alongside
    __call__ in a way static type checking understands.
    """

    def __init__(self, model: AnticipateGCN, device: torch.device):
        self._model = model
        self._device = device

    def __call__(self, datum: SceneGraph) -> float:
        gcn_data = prepare_gcn_input((datum, -1))
        if gcn_data.x is not None:
            gcn_data.batch = torch.zeros(gcn_data.x.size(0), dtype=torch.long)

            with torch.no_grad():
                out = self._model.forward(convert_batch_format(gcn_data), self._device)
                out = out[:, 0].detach().cpu().numpy()
                return out[0]
        # if an invalid graph nodes features vector was passed in
        return -1

    def batch(self, data: list[SceneGraph]) -> list[float]:
        """
        Batched form of __call__: evaluates every scene graph in `data` with a
        single model call instead of one call per graph. Returns one value per
        input graph, in the same order (-1 for an invalid graph, same as
        __call__'s single-graph case).
        """
        items = [prepare_gcn_input((datum, -1)) for datum in data]
        valid = [(idx, item) for idx, item in enumerate(items) if item.x is not None]
        results: list[float] = [-1] * len(data)
        if not valid:
            return results

        indices, valid_items = zip(*valid)
        batch = Batch.from_data_list(list(valid_items))
        with torch.no_grad():
            out = self._model.forward(convert_batch_format(batch), self._device)
            out = out[:, 0].detach().cpu().numpy()
        for idx, value in zip(indices, out):
            results[idx] = value
        return results
