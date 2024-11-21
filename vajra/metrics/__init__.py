# Vayuvahana Technologies Private Limited Vajra, AGPL-3.0 License

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
from vajra.utils import threaded, TryExcept, StringOps, LOGGER
from vajra.plotting import plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)

__all__ = {"functions":["fitness", "smooth", 
                        "ap_per_class", "compute_ap", "box_iou", 
                        "bbox_iou", "bbox_ioa", "plot_pr_curve", 
                        "plot_mc_curve", "mask_iou", "keypoint_iou", 
                        "_get_cov_mat", "probabilistic_iou",
                        "batch_probabilistic_iou", "smooth_bce"],
            "classes": ["Metrics", "DetectionMetrics", 
                        "SegmentationMetrics", "PosMetrics", 
                        "OrientedBboxMetrics"]
          }

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def mask_iou(mask1, mask2, eps=1e-7):
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection
    return intersection / (union + eps)

def keypoint_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    dimensions = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)
    kpt_mask = kpt1[..., 2] != 0
    e = dimensions / (2 * sigma).pow(2) / (area[:, None, None] + eps) / 2
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)

def _get_cov_mat(boxes):
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

def probabilistic_iou(obb1, obb2, CIoU=False, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_cov_mat(obb1)
    a2, b2, c2 = _get_cov_mat(obb2)
    
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2/h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha # CIoU
    return iou

def batch_probabilistic_iou(obb1, obb2, eps = 1e-7):
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_cov_mat(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_cov_mat(obb2))
    
    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd

def smooth_bce(eps=0.1):
    return 1.0 - 0.5*eps, 0.5*eps

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, num_classes, conf=0.25, iou_thres=0.45, task="detect"):
        self.task = task
        self.matrix = np.zeros((num_classes + 1, num_classes + 1)) if self.task == "detect" else np.zeros((num_classes, num_classes))
        self.nc = num_classes
        self.conf = 0.25 if conf in {None, 0.001} else conf
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probabilistic_iou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1 # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class

    @TryExcept('WARNING! ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=(), on_plot=None):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ['background']) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           "size": 8},
                       cmap='Blues',
                       fmt='.2f' if normalize else ".0f",
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + "Normalized" * normalize
        ax.set_ylabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig_name = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(fig_name, dpi=250)
        plt.close(fig)

        if on_plot:
            on_plot(fig_name)

    def print(self):
        for i in range(self.nc + 1):
            LOGGER.info(' '.join(map(str, self.matrix[i])))
            

class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''
    
    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)
    
    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()
    
    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MDPIoU=False, feat_h=640, feat_w=640, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
        return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU
    return iou  # IoU


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(nx4)
    box2:       np.array of shape(mx4)
    returns:    np.array of shape(nxm)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)

class Metrics(StringOps):
    def __init__(self) -> None:
        self.precision = []
        self.recall = []
        self.f1_metric = []
        self.all_ap = []
        self.ap_class_idx = []
        self.num_classes = 0
    
    @property
    def ap(self):
        return self.all_ap.mean(1) if len(self.all_ap) else []
    
    @property
    def ap50(self):
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def mp(self):
        return self.precision.mean() if len(self.precision) else float(len(self.all_ap))

    @property
    def mr(self):
        return self.recall.mean() if len(self.recall) else float(len(self.all_ap))
    
    @property
    def map50(self):
        return self.all_ap[:, 0].mean() if len(self.all_ap) else float(len(self.all_ap))

    @property
    def map75(self):
        return self.all_ap[:, 5].mean() if len(self.all_ap) else float(len(self.all_ap))

    @property
    def map(self):
        return self.all_ap.mean() if len(self.all_ap) else float(len(self.all_ap))
    
    def means(self):
        return [self.mp, self.mr, self.map50, self.map]

    def class_aware_results(self, i):
        return self.precision[i], self.recall[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        maps = np.zeros(self.num_classes) + self.map
        for i, c in enumerate(self.ap_class_idx):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        weights = [0.0, 0.0, 0.1, 0.9] # Precision, Recall, mAP@0.50, mAP@0.95
        return (np.array(self.means()) * weights).sum()

    def update(self, results):
        (self.precision, self.recall, self.f1_metric, 
         self.all_ap, self.ap_class_idx, self.precision_curve, self.recall_curve,
         self.f1_curve, self.px, self.prec_values) = results

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.precision_curve, "Confidence", "Precision"],
            [self.px, self.recall_curve, "Confidence", "Recall"],
        ]

class DetectionMetrics(StringOps):
    def __init__(self, save_dir=Path("."), plot=False, names=(), on_plot=None) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.names = names
        self.bbox = Metrics()
        self.on_plot = on_plot
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = 'detect'

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(tp, conf, pred_cls, target_cls, plot=self.plot,
                               save_dir=self.save_dir, names=self.names, 
                               on_plot=self.on_plot)[2:]
        self.bbox.num_classes = len(self.names)
        self.bbox.update(results)

    @property
    def keys(self):
        return ["metrics/precision(Box)", "metrics/recall(Box)", "metrics/mAP50(Box)", "metrics/mAP50-95(Box)"]

    def means(self):
        return self.bbox.means()

    def class_aware_results(self, i):
        return self.bbox.class_aware_results(i)

    @property
    def maps(self):
        return self.bbox.maps

    @property
    def fitness(self):
        return self.bbox.fitness()

    @property
    def ap_class_index(self):
        return self.bbox.ap_class_idx

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.means() + [self.fitness]))

    @property
    def curves(self):
        return ['Precision/Recall(Box)', 'F1-Confidence(Box)', 'Precision-Confidence(Box)', 'Recall-Confidence(Box)']

    @property
    def curves_results(self):
        return self.bbox.curves_results

class SegmentationMetrics(StringOps):
    def __init__(self, save_dir=Path("."), plot=False, names=(), on_plot=None) -> None:
        
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.bbox = Metrics()
        self.seg = Metrics()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = 'segmentation'

    def process(self, tp, tp_mask, conf, pred_cls, target_cls):
        results_mask = ap_per_class(tp_mask, conf, pred_cls, target_cls,
                                    plot=self.plot, on_plot=self.on_plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix='Mask')[2:]
        self.seg.num_classes = len(self.names)

        self.seg.update(results_mask)
        results_bbox = ap_per_class(tp, conf, pred_cls, target_cls,
                                    plot=self.plot, on_plot=self.on_plot,
                                    save_dir=self.save_dir, names=self.names,
                                    prefix='Bbox')[2:]
        self.bbox.num_classes = len(self.names)
        self.bbox.update(results_bbox)

    @property
    def keys(self):
        return [
            "metrics/precision(Box)",
            "metrics/recall(Box)",
            "metrics/mAP50(Box)",
            "metrics/mAP50-95(Box)",
            "metrics/precision(Mask)",
            "metrics/recall(Mask)",
            "metrics/mAP50(Mask)",
            "metrics/mAP50-95(Mask)",
        ]
    
    def means(self):
        return self.bbox.means() + self.seg.means()

    def class_aware_results(self, i):
        return self.bbox.class_aware_results(i) + self.seg.class_aware_results(i)

    @property
    def maps(self):
        return self.bbox.maps + self.seg.maps

    @property
    def fitness(self):
        return self.seg.fitness() + self.bbox.fitness()

    @property
    def ap_class_index(self):
        return self.bbox.ap_class_idx

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], self.means() + [self.fitness]))

    @property
    def curves(self):
        return [
            "Precision-Recall(Box)",
            "F1-Confidence(Box)",
            "Precision-Confidence(Box)",
            "Recall-Confidence(Box)",
            "Precision-Recall(Mask)",
            "F1-Confidence(Mask)",
            "Precision-Confidence(Mask)",
            "Recall-Confidence(Mask)"
        ]

    @property
    def curves_results(self):
        return self.bbox.curves_results + self.seg.curves_results

class PoseMetrics(SegmentationMetrics):
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.bbox = Metrics()
        self.pose = Metrics()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = 'pose-detection'

    def process(self, tp, tp_p, conf, pred_cls, target_cls):
        results_pose = ap_per_class(
            tp_p, conf, pred_cls, target_cls,
            plot=self.plot, on_plot=self.on_plot,
            save_dir=self.save_dir, names=self.names,
            prefix='Pose'
        )[2:]
        self.pose.num_classes = self.bbox.num_classes = len(self.names)
        self.pose.update(results_pose)

        results_bbox = ap_per_class(
            tp, conf, pred_cls, target_cls,
            plot=self.plot, on_plot=self.on_plot,
            save_dir=self.save_dir, names=self.names,
            prefix='Bbox'
        )[2:]
        self.bbox.update(results_bbox)

    @property
    def keys(self):
        return [
            "metrics/precision(Box)",
            "metrics/recall(Box)",
            "metrics/mAP50(Box)",
            "metrics/mAP50-95(Box)",
            "metrics/precision(Pose)",
            "metrics/recall(Pose)",
            "metrics/mAP50(Pose)",
            "metrics/mAP50-95(Pose)"
        ]

    def means(self):
        return self.bbox.means() + self.pose.means()

    def class_aware_results(self, i):
        return self.bbox.class_aware_results(i) + self.pose.class_aware_results(i)

    @property
    def maps(self):
        return self.bbox.maps + self.pose.maps

    @property
    def fitness(self):
        return self.bbox.fitness() + self.pose.fitness()

    @property
    def curves(self):
        return[
            'Precision-Recall(Box)',
            'F1-Confidence(Box)',
            'Precision-Confidence(Box)',
            'Recall-Confidence(Box)',
            'Precision-Recall(Pose)',
            'F1-Confidence(Pose)',
            'Precision-Confidence(Pose)',
            'Recall-Confidence(Pose)'
        ]

    @property
    def curves_results(self):
        return self.bbox.curves_results + self.pose.curves_results

class ClassificationMetrics(StringOps):
    def __init__(self) -> None:
        self.top1 = 0
        self.top5 = 0
        self.speed = {'preprocess': 0.0, 'inference':0.0, 'loss':0.0, 'postprocess': 0.0}
        self.task = 'classification'

    def process(self, targets, preds):
        preds, targets = torch.cat(preds), torch.cat(targets)
        correct = (targets[:, None] == preds).float()
        accuracy = torch.stack((correct[:, 0], correct.max(1).values), dim=1)
        self.top1, self.top5 = accuracy.mean(0).tolist() 

    @property
    def fitness(self):
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return []

class MultiLabelClassificationMetrics(StringOps):

    def __init__(self) -> None:
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "multilabel_classify"
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def process(self, targets, pred):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        batch_targets = targets[0]
        batch_pred = pred[0]
        for i in range(0, len(batch_pred) - 1):
            for j in range(0, len(batch_pred[i]) - 1):
                if batch_targets[i][j] == 1:
                    if batch_pred[i][j] >= 0.5:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    if batch_pred[i][j] <= 0.5:
                        true_negatives += 1
                    else:
                        false_negatives += 1
        self.precision = true_positives / (true_positives + false_positives)
        self.recall = true_positives / (true_positives + false_negatives)
        try:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        except:
            LOGGER.exception("Error in calculating F1 Score.")
            self.f1 = 0
        LOGGER.info(f"Precision: {self.precision}, Recall: {self.recall}, F1 Score: {self.f1}")

    @property
    def fitness(self):
        return (self.precision + self.recall + self.f1) / 3

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], [self.precision, self.recall, self.f1, self.fitness]))

    @property
    def keys(self):
        return ["metrics/precision", "metrics/recall", "metrics/f1"]

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return []

class OrientedBboxMetrics(StringOps):
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.bbox = Metrics()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=self.plot,
            save_dir=self.save_dir, names=self.names,
            on_plot=self.on_plot
        )[2:]
        self.bbox.num_classes = len(self.names)
        self.bbox.update(results)

    @property
    def keys(self):
        return ["metrics/precision(Box)", "metrics/recall(Box)", "metrics/mAP50(Box)","metrics/mAP50-95(Box)"]
    
    def means(self):
        return self.bbox.means()
    
    def class_aware_results(self, i):
        return self.bbox.class_aware_results(i)

    @property
    def maps(self):
        return self.bbox.maps

    @property
    def fitness(self):
        return self.bbox.fitness()

    @property
    def ap_class_index(self):
        return self.bbox.ap_class_idx

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], self.means() + [self.fitness]))

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return []