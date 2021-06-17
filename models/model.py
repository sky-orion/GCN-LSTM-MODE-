import torch
import torch.nn as nn
import torch.nn.functional as F

from models import layer
from models import attention
from models.attention import ConvLSTMCell, AttentionStem


class AirspaceModel(nn.Module):
    def __init__(self, 
                 in_channels=17, 
                 out_channels=1,
                 residual_channels=32, 
                 dilation_channels=32,
                 hidden_channels=128,
                skip_channels=256,
                 end_channels=128, 
                 supports=None,
                 drop_rate=0.3,
                 kernel_size=2, 
                 blocks=4,
                 layers=2,
                 n_vertex=126,
                 use_graph_conv=True,
                 adaptive_mat_init=None,
                 adaptive_mat_size=10,
                 handle_minor_features=True,
                 use_adaptive_mat_only=False,
                 device='cpu',
                 seqlen=24):
        super(AirspaceModel, self).__init__()
        self.hidden_channels=hidden_channels
        self.blocks = blocks
        self.layers = layers
        self.drop_rate = drop_rate
        self.handle_minor_features = handle_minor_features
        self.supports = supports or []
        self.supports_len = len(self.supports)
        self.use_graph_conv = use_graph_conv
        self.adaptive_mat_init = adaptive_mat_init
        self.use_adaptive_mat_only = use_adaptive_mat_only

        self.lstm=nn.ModuleList()
        self.device=device
        self.nodes=n_vertex
        self.lstminput = in_channels#feature数目
        self.lstmhidden = 100
        self.lstmnum_layers = 2
        self.seqlen = seqlen
        # self.depth=self.blocks=self.layers
        self.linear = nn.Linear(in_features=self.lstmhidden, out_features=1).to(device)
        # self.linearx = nn.Linear(in_features=((self.seqlen-self.depth-2) if (self.seqlen-self.depth-2)>0 else 1), out_features=1).to(device)
        for _ in range(self.nodes):
            self.lstm.append(nn.LSTM(input_size=self.lstminput, hidden_size=self.lstmhidden,
                            num_layers=self.lstmnum_layers,
                            bidirectional=False).to(self.device))
        self.lstmendconv =nn.Conv2d(self.seqlen, out_channels, 1).to(self.device)


        receptive_field = 1
        depth = list(range(blocks * layers))
        if self.use_graph_conv and self.adaptive_mat_init is not None:
            nodevecs = self.svd_init(adaptive_mat_size, self.adaptive_mat_init)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs]

        self.supports_len = self.supports_len if not use_adaptive_mat_only else 1

        if self.handle_minor_features:
            self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1,1))
            self.minor_features_conv = nn.Conv2d(in_channels-1, residual_channels, kernel_size=(1,1))
        else:
            self.start_conv = nn.Conv2d(in_channels, residual_channels,  kernel_size=(1,1))
        # 1x1 convolution for residual connection
        self.residual_convs = nn.ModuleList([nn.Conv1d(dilation_channels, residual_channels,  kernel_size=(1,1)) for _ in depth])
        # 1x1 convolution for skip connection
        # AttentionStem
        self.skip_convs = nn.ModuleList([nn.Conv1d(dilation_channels, skip_channels, kernel_size=(1,1)) for _ in depth])
        self.bn = nn.ModuleList([nn.BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = nn.ModuleList([layer.DiffGraphConv(dilation_channels, residual_channels, drop_rate,
                                                              self.supports_len) for _ in depth])
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.attentionconv=AttentionStem(residual_channels, dilation_channels,kernel_size= 1)

        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                # dilated convolutions
                # AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                self.gate_convs.append(nn.Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.output_layer = layer.OutputLayer(skip_channels, end_channels, out_channels)
        # self.output_layer1 = nn.Conv2d(dilation_channels, skip_channels, 1)
        self.receptive_field = receptive_field

    @staticmethod
    def svd_init(adp_mat_size, adp_mat_init):
        m, p, n = torch.svd(adp_mat_init)
        nodevec1 = torch.mm(m[:, :adp_mat_size], torch.diag(p[:adp_mat_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:adp_mat_size] ** 0.5), n[:, :adp_mat_size].t())
        return nodevec1, nodevec2

    def forward(self, x,hnlist=[],cnlist=[]):
        # Input shape is (batch_size, seq_len, n_vertex, features)
        global lstmlayer
        if (len(hnlist)==0 and len(cnlist)==0 ):
            for i in range(self.nodes):
                hnlist.append(torch.randn(self.lstmnum_layers, x.size(0), self.lstmhidden).to(self.device))
                cnlist.append(torch.randn(self.lstmnum_layers, x.size(0), self.lstmhidden).to(self.device))
        newhnlist=[]
        newcnlist=[]
        for i in range(x.size(2)):
            # print(x.size(),x[:, :, i, :].permute(1, 0, 2).size())
            # print((x[:, :, i, :].permute(1, 0, 2)),(hnlist[i].to(self.device),cnlist[i].to(self.device)))
            lstm_out, (tmphn,tmpcn) = self.lstm[i]((x[:, :, i, :].permute(1, 0, 2)),(hnlist[i],cnlist[i]))
            newhnlist.append(tmphn)
            newcnlist.append(tmpcn)
            linear_out = self.linear(lstm_out)
            # print(x.size(),x[:, :, i, :].permute(1, 0, 2).size(),lstm_out.size(),linear_out.unsqueeze(0).size())
            if (i == 0):
                lstmlayer=linear_out.unsqueeze(0)
            else:
                lstmlayer=torch.cat((lstmlayer,linear_out.unsqueeze(0)), 0)
        lstmlayer=lstmlayer.permute(2,1,0,3)
        lstmlayer=self.lstmendconv(lstmlayer)

        x = x.transpose(1,3)
        seq_len = x.size(3)
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))
        # print(x.size(),self.receptive_field)
        if self.handle_minor_features:
            x = self.start_conv(x[:,-1,...].unsqueeze(1)) + F.leaky_relu(self.minor_features_conv(x[:,:-1,...]))
        else:
            x = self.start_conv(x)
        skip = 0

        if self.use_adaptive_mat_only:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj_mats = [adp]
        elif self.adaptive_mat_init is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj_mats = self.supports + [adp]
        else:
            adj_mats = self.supports
        # print("origin", x.size())
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            x = torch.mul(torch.tanh(self.filter_convs[i](residual)), torch.sigmoid(self.gate_convs[i](residual)))
            #gate
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):
                break

            if self.use_graph_conv:
                x = x + self.graph_convs[i](x, adj_mats)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]#取最后几个数

            x = self.bn[i](x)
            # print("inloop",x.size())
        # print("before", x.size())
        x = F.relu(skip)
        # print(x.size(), lstmlayer.size())
        x = self.output_layer(x)# downsample to (batch_size, seq_len, n_vertex, features)
        # print(x.size(),lstmlayer.size())
        out = lstmlayer + x
        return out,newhnlist,newcnlist
seqlen=30
temp = torch.randn((16, seqlen, 126, 17))
linearx = AirspaceModel(out_channels=seqlen,seqlen=seqlen,blocks=5,layers=3)
hn=[]
cn=[]
# x = F.softmax(temp,dim=2)
print("input",temp.size())
out,hn,cn=linearx(temp)
print("output",out.size(),linearx.receptive_field)
# model =AirspaceModel(in_channels=17)
# x,y,z=model(temp)
# # x,y=model(temp,y)
# print(x.size())
# x,y,z=model(temp,y,z)
# inp = torch.rand(1, 3, 3, 3)
# inp_unf = torch.nn.functional.unfold(inp, (2, 2), stride=1)
# inp_unf=inp.unfold(2, 2,1)
# print(inp.size())
# print(inp)
# print(inp_unf.size())
# print(inp_unf)

# refold = torch.nn.functional.fold(inp_unf, output_size=(3, 3), kernel_size=(2, 2), stride=1)
# print(refold.size())
# print(refold)

