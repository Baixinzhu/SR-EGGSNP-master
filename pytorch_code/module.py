import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    #input_size输入的特征维度，对应每个时间步的特征数，hidden_size为神经元个数
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # 输入门
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        # 遗忘门
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # 输出门
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        # 单元状态
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        #对Pytorch张量x的大小进行解包，x.size()返回x的大小。以元组的形式
        bs, _ = x.size()

        h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # 输入门
        i_t = torch.sigmoid(x @ self.W_ii.t() + self.b_ii + h_t @ self.W_hi.t() + self.b_hi)

        # 遗忘门
        f_t = torch.sigmoid(x @ self.W_if.t() + self.b_if + h_t @ self.W_hf.t() + self.b_hf)

        # 输出门
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho)

        # 单元状态
        g_t = torch.tanh(x @ self.W_ig.t() + self.b_ig + h_t @ self.W_hg.t() + self.b_hg)

        # 更新单元状态
        c_t = f_t * c_t + i_t * g_t

        # 更新隐藏状态
        h_t = o_t * torch.tanh(c_t)
        h_t = self.dropout(h_t)

        return h_t, c_t

class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        # 重置门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size))

        # 更新门
        self.W_iz = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hz = nn.Parameter(torch.Tensor(hidden_size))

        # 隐藏状态
        self.W_in = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        bs, _ = x.size()

        h_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        # 重置门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h_t @ self.W_hr.t() + self.b_hr)

        # 更新门
        z_t = torch.sigmoid(x @ self.W_iz.t() + self.b_iz + h_t @ self.W_hz.t() + self.b_hz)

        # 隐藏状态的更新
        n_t = torch.tanh(x @ self.W_in.t() + self.b_in + r_t * (h_t @ self.W_hn.t() + self.b_hn))

        # 更新隐藏状态
        h_t = (1 - z_t) * h_t + z_t * n_t
        h_t = self.dropout(h_t)
        return h_t

# 使用自定义的LSTM和GRU单元构建完整的模型
# output_size=chunk_size在隐藏状态中引入分块结构，控制隐层的分块大小
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(CustomLSTM, self).__init__()
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len,_ = x.size()
        h_t, c_t = torch.zeros(self.lstm_cell.hidden_size).to(x.device), torch.zeros(self.lstm_cell.hidden_size).to(x.device)

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t , :], (h_t, c_t))

        out = self.linear(h_t)
        return out

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(CustomGRU, self).__init__()
        self.gru_cell = CustomGRUCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len, _ = x.size()
        h_t = torch.zeros(self.gru_cell.hidden_size).to(x.device)

        for t in range(seq_len):
            h_t = self.gru_cell(x[:, t, :], h_t)

        out = self.linear(h_t)
        return out


class EGSNPCell(nn.Module):
    #input_size输入的特征维度，对应每个时间步的特征数，hidden_size为神经元个数
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(EGSNPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # 输入门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ur = nn.Parameter(torch.Tensor(hidden_size))


        # 消耗门
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uc = nn.Parameter(torch.Tensor(hidden_size))

        # 候选单元状态
        self.W_ia = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_ia = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ua = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        #对Pytorch张量x的大小进行解包，x.size()返回x的大小。以元组的形式
        bs, _ = x.size()

        h_t, u_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # 输入门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h_t @ self.W_hr.t() + self.b_hr + u_t @ self.W_ur + self.b_ur)

        # 消耗门
        c_t = torch.sigmoid(x @ self.W_ic.t() + self.b_ic + h_t @ self.W_hc.t() + self.b_hc + u_t @ self.W_uc + self.b_uc)

        #候选状态（脉冲更新公式）
        a = torch.tanh(x @ self.W_ia.t() + self.b_ia + u_t @ self.W_ua.t() + self.b_ua)
        # 更新内部神经元状态

        u_t = r_t * u_t - c_t * a
        # 更新隐藏状态
        h_t = a
        h_t = self.dropout(h_t)
        return h_t, u_t
class EGSNP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(EGSNP, self).__init__()
        self.EGSNP_cell = EGSNPCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len,_ = x.size()
        h_t, u_t = torch.zeros(self.EGSNP_cell.hidden_size).to(x.device), torch.zeros(self.EGSNP_cell.hidden_size).to(x.device)

        for t in range(seq_len):
            h_t, u_t = self.EGSNP_cell(x[:, t , :], (h_t, u_t))

        out = self.linear(h_t)
        return out


class LSTMSNPCell(nn.Module):
    # input_size输入的特征维度，对应每个时间步的特征数，hidden_size为神经元个数
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(LSTMSNPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        # 输入门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ur = nn.Parameter(torch.Tensor(hidden_size))

        # 遗忘门
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uc = nn.Parameter(torch.Tensor(hidden_size))

        # 输出门
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_uo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uo = nn.Parameter(torch.Tensor(hidden_size))

        # 单元状态
        self.W_ia = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ia = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ua = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        # 对Pytorch张量x的大小进行解包，x.size()返回x的大小。以元组的形式
        bs, _ = x.size()

        h_t, u_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # 输入门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + u_t @ self.W_ur.t() + self.b_ur)

        # 遗忘门
        c_t = torch.sigmoid(x @ self.W_ic.t() + self.b_ic + u_t @ self.W_uc.t() + self.b_uc)
        # 输出门
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + u_t @ self.W_uo.t() + self.b_uo)
        # 候选状态（脉冲更新公式）
        a = torch.tanh(x @ self.W_ia.t() + self.b_ia + u_t @ self.W_ua.t() + self.b_ua)
        # 更新内部神经元状态

        u_t = r_t * u_t - c_t * a
        # 更新隐藏状态
        h_t = o_t * a
        h_t = self.dropout(h_t)
        return h_t, u_t


class LSTMSNP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(LSTMSNP, self).__init__()
        self.LSTMSNP_cell = LSTMSNPCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len, _ = x.size()
        h_t, u_t = torch.zeros(self.LSTMSNP_cell.hidden_size).to(x.device), torch.zeros(self.LSTMSNP_cell.hidden_size).to(
            x.device)

        for t in range(seq_len):
            h_t, u_t = self.LSTMSNP_cell(x[:, t, :], (h_t, u_t))

        out = self.linear(h_t)
        return out

class NSNPAUCell(nn.Module):
    # input_size输入的特征维度，对应每个时间步的特征数，hidden_size为神经元个数
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(NSNPAUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        # 输入门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ur = nn.Parameter(torch.Tensor(hidden_size))

        # 遗忘门
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uc = nn.Parameter(torch.Tensor(hidden_size))

        # 输出门
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_uo = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uo = nn.Parameter(torch.Tensor(hidden_size))

        # 单元状态
        self.W_ia = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ia = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ua = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        # 对Pytorch张量x的大小进行解包，x.size()返回x的大小。以元组的形式
        bs, _ = x.size()

        h_t, u_t = (torch.zeros(self.hidden_size).to(x.device),
                    torch.zeros(self.hidden_size).to(x.device)) if init_states is None else init_states

        # 输入门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h_t @ self.W_hr.t() + self.b_hr + u_t @ self.W_ur + self.b_ur)

        # 遗忘门
        c_t = torch.sigmoid(x @ self.W_ic.t() + self.b_ic + h_t @ self.W_hc.t() + self.b_hc + u_t @ self.W_uc + self.b_uc)
        # 输出门
        o_t = torch.sigmoid(x @ self.W_io.t() + self.b_io + h_t @ self.W_ho.t() + self.b_ho + u_t @ self.W_uo + self.b_uo)
        # 候选状态（脉冲更新公式）
        a = torch.tanh(x @ self.W_ia.t() + self.b_ia + u_t @ self.W_ua.t() + self.b_ua)
        # 更新内部神经元状态

        u_t = r_t * u_t - c_t * a
        # 更新隐藏状态
        h_t = o_t * a
        h_t = self.dropout(h_t)
        return h_t, u_t


class NSNPAU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(NSNPAU, self).__init__()
        self.NSNPAU_cell = NSNPAUCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len, _ = x.size()
        h_t, u_t = torch.zeros(self.NSNPAU_cell.hidden_size).to(x.device), torch.zeros(self.NSNPAU_cell.hidden_size).to(
            x.device)

        for t in range(seq_len):
            h_t, u_t = self.NSNPAU_cell(x[:, t, :], (h_t, u_t))

        out = self.linear(h_t)
        return out

class GSNPCell(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate=0.0):
        super(GSNPCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        # 重置门
        self.W_ir = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ur = nn.Parameter(torch.Tensor(hidden_size))

        # 更新门
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_uc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ic = nn.Parameter(torch.Tensor(hidden_size))
        self.b_uc = nn.Parameter(torch.Tensor(hidden_size))

        # 隐藏状态
        self.W_ia = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ia = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ua = nn.Parameter(torch.Tensor(hidden_size))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states=None):
        bs, _ = x.size()

        u_t = torch.zeros(self.hidden_size).to(x.device) if init_states is None else init_states

        # 重置门
        r_t = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + u_t @ self.W_ur.t() + self.b_ur)

        # 更新门
        c_t = torch.sigmoid(x @ self.W_ic.t() + self.b_ic + u_t @ self.W_uc.t() + self.b_uc)

        # 候选隐藏状态的更新

        a = torch.tanh(x @ self.W_ia.t() + self.b_ia + u_t @ self.W_ua.t() + self.b_ua)
        # 更新隐藏状态
        u_t = r_t * u_t + c_t * a
        u_t = self.dropout(u_t)
        return u_t

class GSNP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate=0.0):
        super(GSNP, self).__init__()
        self.GSNP_cell = GSNPCell(input_size, hidden_size,dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bs, seq_len, _ = x.size()
        u_t = torch.zeros(self.GSNP_cell.hidden_size).to(x.device)

        for t in range(seq_len):
            u_t = self.GSNP_cell(x[:, t, :], u_t)

        out = self.linear(u_t)
        return out