import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models import layer
import math


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1,bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,dilation=dilation ,bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=dilation ,bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, dilation=dilation ,bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        # print(k_out.size(),torch.nn.functional.unfold(k_out,kernel_size=(3,3),stride=1).size())
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # print(k_out.size())
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        print(k_out.size())
        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        print(k_out.size(), q_out.size(),torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).size())
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = config['num_of_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return output


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=1):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = ((kernel_size - 1) // 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        print(x.size(),h.size(),c.size())
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda())


class tpalstm(nn.Module):
    def __init__(self, args, data):
        super(tpalstm, self).__init__()
        self.use_cuda = args.cuda
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = data.original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.hidden_state_features_uni_lstm = args.hidden_state_features_uni_lstm
        self.attention_size_uni_lstm = args.attention_size_uni_lstm
        self.num_layers_uni_lstm = args.num_layers_uni_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.uni_lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_state_features_uni_lstm,
                                num_layers=args.num_layers_uni_lstm,
                                bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'

        self.attention_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns, requires_grad=True)).cuda()
        self.context_vector_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()
        self.final_hidden_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()
        self.final_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()

        self.bridge_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True)).cuda()

        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        torch.nn.init.xavier_uniform(self.attention_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.context_vector_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.final_hidden_uni_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.bridge_matrix)

        self.conv1 = nn.Conv2d(1, self.hidC,
                               kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x.cuda()

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        ##Incase in future bidirectional lstms are to be used, size of hn would needed to be modified a little (as output is of size (num_layers * num_directions, batch, hidden_size))
        input_to_lstm = x.permute(1, 0,
                                  2).contiguous()  # input to lstm is of shape (seq_len, batch, features) (x is of shape (batch_size, seq_length, features))
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features);
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer));
        convolution_output = self.dropout(convolution_output);

        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        input = torch.zeros(self.attention_matrix.size(0), x.size(1), x.size(2))
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            input[:x.size(0), :, :] = x
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output
            input = x.cuda()

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring.cuda()
        final_hn_realigned = final_hn_realigned.cuda()
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        scoring_function = torch.bmm(mat1, final_hn_realigned).cuda()
        alpha = torch.nn.functional.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1).cuda()

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix,
                                                                                            context_vector)

        """
            Up until now TPA-LSTM has been implemented in pytorch

            Modification
            Background: TPA-LSTM guys use all features together and stack them up during the RNN stage. This treats one time step as one entity, killing the individual
            properties of each variable. At the very end, the variable we are predicting for must depend on itself the most.

            Proposition: Use lstm on each variable independently (assuming them to be independent). Using the hidden state of each time step, along with the hidden state
            of the CNN, now apply the same attention model. This method will preserve the identiy of the individual series by not considering all of them as one state.   
        """
        individual_all_hidden_states = None
        individual_last_hidden_state = None
        for feature_num in range(0, self.original_columns):
            individual_feature = input[:, :, feature_num].view(input.size(0), input.size(1), -1).permute(1, 0,
                                                                                                         2).contiguous().cuda()
            uni_output, (uni_hn, uni_cn) = self.uni_lstm(
                individual_feature)  # Output of hn is of the size  (num_layers * num_directions, batch, hidden_size)  |  num_layers = 2 in bidirectional lstm
            if (feature_num == 0):
                individual_all_hidden_states = uni_output.permute(1, 0, 2).contiguous()
                individual_last_hidden_state = uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1, 0,
                                                                                                          2).contiguous()
            else:
                individual_all_hidden_states = torch.cat(
                    (individual_all_hidden_states, uni_output.permute(1, 0, 2).contiguous()), 1)
                individual_last_hidden_state = torch.cat((individual_last_hidden_state,
                                                          uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1,
                                                                                                                     0,
                                                                                                                     2).contiguous()),
                                                         1)

        ## *****************DIMENSIONS OF individual_all_hidden_states are (batch_size, time series length/window size, hidden_state_features, total univariate series)*****************##
        ## *****************DIMENSIONS OF individual_last_hidden_state are (batch_size, 1, hidden_state_features, total univariate series)*****************##

        individual_all_hidden_states = individual_all_hidden_states.view(input.size(0), input.size(1),
                                                                         self.hidden_state_features_uni_lstm, -1).cuda()
        individual_last_hidden_state = individual_last_hidden_state.view(input.size(0), 1,
                                                                         self.hidden_state_features_uni_lstm, -1).cuda()
        """
        Calculate the attention score for all of these
        """
        univariate_attended = []
        h_output = None
        for feature_num in range(0, self.original_columns):
            attention_matrix_uni = self.attention_matrix_uni_lstm[:, :, :, feature_num]
            context_vector_matrix_uni = self.context_vector_matrix_uni_lstm[:, :, :, feature_num]
            hidden_matrix = self.final_hidden_uni_matrix[:, :, :, feature_num]
            final_matrix = self.final_uni_matrix[:, :, :, feature_num]
            all_hidden_states_single_variable = individual_all_hidden_states[:, :, :, feature_num].cuda()
            final_hidden_state = individual_last_hidden_state[:, :, :, feature_num].permute(0, 2, 1).contiguous().cuda()

            mat1 = torch.bmm(all_hidden_states_single_variable, attention_matrix_uni).cuda()
            mat2 = torch.bmm(mat1, final_hidden_state).cuda()
            attention_score = torch.sigmoid(mat2).cuda()

            context_vector_individual = attention_score * all_hidden_states_single_variable
            context_vector_individual = torch.sum(context_vector_individual, dim=1).cuda()
            context_vector_individual = context_vector_individual.view(context_vector_individual.size(0),
                                                                       context_vector_individual.size(1), 1).cuda()

            attended_states = torch.bmm(context_vector_matrix_uni, context_vector_individual).cuda()
            h_intermediate1 = attended_states + torch.bmm(hidden_matrix, final_hidden_state).cuda()

            if (feature_num == 0):
                h_output = torch.bmm(final_matrix, h_intermediate1).cuda()
            else:
                h_output += torch.bmm(final_matrix, h_intermediate1).cuda()

        h_intermediate2 = torch.bmm(self.bridge_matrix, h_output).cuda()

        """
           Combining the two
        """
        h_intermediate = h_intermediate + h_intermediate2
        result = torch.bmm(self.final_matrix, h_intermediate.cuda())
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """
           Remove from result the extra result points which were added as a result of padding
        """
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """

        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.original_columns);
            res = final_result + z;

        return torch.sigmoid(res)
# x = torch.rand((10,32, 126, 17))
# x=x[:,:,0,:].permute((1,0,2))
# print(x.size())
#
# lstm=nn.LSTM(input_size=17, hidden_size=10,
#                             num_layers=1,
#                             bidirectional=False)
# input_to_lstm = x.permute(0, 1, 2).contiguous()
# print(input_to_lstm.size())# input to lstm is of shape (seq_len, batch, features) (x is of shape (batch_size, seq_length, features))
# h_all=torch.rand(1, 10, 10)
# c_all=torch.rand(1, 10, 10)
# a=(h_all,c_all)
# # print(a[0],a[1])
# lstm_hidden_states, (h_all, c_all) = lstm(input_to_lstm,a)
# linear = nn.Linear(in_features=10, out_features=1)
# tmp=x.unsqueeze(0)
# tmp=torch.cat((tmp, x.unsqueeze(0)), 0)
# tmp.permute(2,1,0,3)
# print(tmp.size())
# print(lstm_hidden_states.size(),h_all.size(),c_all.size(),linear(lstm_hidden_states).size())
# for i in range(126):
#     print(i)
# if __name__ == "__main__":
#     config = {
#         "num_of_attention_heads": 2,
#         "hidden_size": 4
#     }
#
#     selfattn = BertSelfAttention(config)
#     print(selfattn)
#     embed_rand = torch.rand((1, 3, 4))
#     print(f"Embed Shape: {embed_rand.shape}")
#     print(f"Embed Values:\n{embed_rand}")
#
#     output = selfattn(embed_rand)
#     print(f"Output Shape: {output.shape}")
#     print(f"Output Values:\n{output}")
# a = torch.arange(16).reshape(4,4)
# print(a.size(),a)
# print(torch.split(a, 2,dim=1))
# # b=ConvLSTMCell(input_channels=32, hidden_channels=128, kernel_size=1)
# print()
# temp = torch.randn((2, 3, 1, 1))
# conv = AttentionStem(3, 3, 1, padding=0)
# print(conv(temp).size())
# print(conv(temp),temp)
