import Parser
import Grid
import Detection
import ChangeType
import order
import move
import route
import route2 as dt1  # dt_min
import route3 as dt2  # dt_all
import numpy as np
from matplotlib import pyplot as plt
import math
import time
import cv2
import seaborn as sns

from matplotlib.widgets import EllipseSelector, RectangleSelector
import matplotlib
matplotlib.use('TkAgg')
from functools import partial

# output image for detecting dense routing area
def prepare_dense_image(net_info, pad_via_info, case, pad_diameter):
    img_canvas = np.zeros((7000, 7000), dtype=np.uint8)
    img_canvas[:, :] = 255
    # for item in pad_via_info:
    #     for pad_vias in item.values():
    #         for pv in pad_vias:
    #             img_canvas = cv2.circle(img_canvas, (int(pv[0])+35000, int(pv[1])+35000), pad_diameter//2, 0, 2)
    for net in net_info:
        # img_canvas = cv2.circle(img_canvas, (int(net[1][0])+35000, int(net[1][1])+35000), pad_diameter//2, 0, 2)
        # img_canvas = cv2.circle(img_canvas, (int(net[2][0])+35000, int(net[2][1])+35000), pad_diameter//2, 0, 2)
        for segment in net[3]:
            img_canvas = cv2.line(
                img_canvas,
                (int(segment[0] / 10) + 3500, int(segment[2] / 10) + 3500),
                (int(segment[1] / 10) + 3500, int(segment[3] / 10) + 3500),
                0,
                1,
            )
    img_canvas = cv2.flip(img_canvas, 0)
    cv2.imwrite(f"../data/{case}/{case}_real_size.png", img_canvas)
    # cv2.imwrite(f"../real_size/{case}.png", img_canvas)

# prepare for the detection image
def prepare_image(net_info, case, pad_diameter):
    plt.figure(dpi=125, figsize=(4.544, 4.544))
    for net in net_info:
        for segment in net[3]:
            plt.plot(
                [segment[0], segment[1]],
                [segment[2], segment[3]],
                color="black",
                linewidth=0.25,
            )
    plt.axis("off")
    plt.axis("square")
    plt.savefig("../data/" + case + "/detection.png", transparent=False)
    img = cv2.imread("../data/" + case + "/detection.png")
    cv2.imwrite("../data/detect/detect.png", img)
    plt.close()

def toggle_selector(event):
    print("Key pressed.")
    if event.key == "t":
        name = type(RectangleSelector).__name__
        if RectangleSelector.active:
            print(f"{name} deactivated.")
            RectangleSelector.set_active(False)
        else:
            print(f"{name} activated.")
            RectangleSelector.set_active(True)

def select_callback(eclick, erelease, coordinates):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    coordinates.extend([x1, y1, x2, y2])
    # findPartialSegments(x1, x2, y1, y2)
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
    print(f"The buttons you used were: {eclick.button} {erelease.button}")

def findPartialNets(net_info, coordinates):
    partial_net_info = []
    print(coordinates)
    for net in net_info:
        # print(f'net: {net[1][0]}, {net[1][1]}, {net[2][0]}, {net[2][1]}')
        for segment in net[3]:
            print(f'{segment[0]}, {segment[1]}, {segment[2]}, {segment[3]}')
            if (coordinatesWithinRectangle(*coordinates, *segment)):
                print(*coordinates, *segment)
                partial_net_info.append(net)
                break
    print(len(net_info))
    print(len(partial_net_info))
    return partial_net_info

def on_segment(p, q, r):
    """Given three colinear points p, q, and r, check if point q lies on segment pr"""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    """Return the orientation of the triplet (p, q, r).
    0 -> p, q and r are colinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def line_intersects(p1, q1, p2, q2):
    """Check if line segment 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False

def line_rectangle_intersection(line_start, line_end, rect_diag1, rect_diag2):
    """Check if the line intersects with the rectangle defined by diagonal points."""
    # Rectangle vertices
    top_left = (min(rect_diag1[0], rect_diag2[0]), max(rect_diag1[1], rect_diag2[1]))
    bottom_right = (max(rect_diag1[0], rect_diag2[0]), min(rect_diag1[1], rect_diag2[1]))
    bottom_left = (top_left[0], bottom_right[1])
    top_right = (bottom_right[0], top_left[1])

    # Check each of the four sides of the rectangle
    if (line_intersects(line_start, line_end, top_left, top_right) or
        line_intersects(line_start, line_end, top_right, bottom_right) or
        line_intersects(line_start, line_end, bottom_right, bottom_left) or
        line_intersects(line_start, line_end, bottom_left, top_left)):
        return True
    return False

def coordinatesWithinRectangle(x1, y1, x2, y2, x1seg, x2seg, y1seg, y2seg):
    line_start = (x1seg, y1seg)
    line_end = (x2seg, y2seg)
    rect_diag1 = (x1, y1)
    rect_diag2 = (x2, y2)
    return line_rectangle_intersection(line_start, line_end, rect_diag1, rect_diag2)

def interactive_tool(net_info, pad_via_info, pad_diameter, coordinates):
    fig, ax = plt.subplots(figsize=(15, 15))

    for item in pad_via_info:
        for pad_vias in item.values():
            for pv in pad_vias:
                # print(f'pad_vias: {pv[0]}, {pv[1]}')
                ax.scatter(
                    pv[0],
                    pv[1],
                    s=(pad_diameter // 2) ** 2 / 200,
                    edgecolors="gray",
                    facecolors="none",
                    linewidths=0.5
                )
    for net in net_info:
        # print(f'net: {net[1][0]}, {net[1][1]}, {net[2][0]}, {net[2][1]}')
        ax.scatter(
            net[1][0],
            net[1][1],
            s=(pad_diameter // 2) ** 2 / 200,
            edgecolors="black",
            facecolors="none",
            linewidths=0.5
        )
        ax.scatter(
            net[2][0],
            net[2][1],
            s=(pad_diameter // 2) ** 2 / 200,
            edgecolors="black",
            facecolors="none",
            linewidths=0.5
        )
        for segment in net[3]:
            ax.plot([segment[0], segment[1]], [segment[2], segment[3]], color="black")
    ax.axis("square")
    callback_with_coords = partial(select_callback, coordinates=coordinates)
    selectors = []
    selectors.append(
        RectangleSelector(
            ax,
            callback_with_coords,
            useblit=True,
            button=[1, 3],  # disable middle button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
    )
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show()
    plt.close()

# plot the routing results
def pic(net_info, pad_via_info, mode, case, pad_diameter, name):
    plt.figure(figsize=(50, 50))
    for item in pad_via_info:
        for pad_vias in item.values():
            for pv in pad_vias:
                plt.scatter(
                    pv[0],
                    pv[1],
                    s=(pad_diameter // 2) ** 2 / 100,
                    edgecolors="gray",
                    facecolors="none",
                )
    for net in net_info:
        plt.scatter(
            net[1][0],
            net[1][1],
            s=(pad_diameter // 2) ** 2 / 100,
            edgecolors="black",
            facecolors="none",
        )
        plt.scatter(
            net[2][0],
            net[2][1],
            s=(pad_diameter // 2) ** 2 / 100,
            edgecolors="black",
            facecolors="none",
        )
        for segment in net[3]:
            plt.plot([segment[0], segment[1]], [segment[2], segment[3]], color="black")
    plt.axis("off")
    plt.axis("square")

    if mode:
        plt.savefig("../data/" + case + "/" + name + ".png")
    else:
        plt.show()
    plt.close()

# main function
def main():

    # user defined parameters
    global_path = "../data/"
    case = input("Case (ex: case_1): ")
    if len(case) == 0:
        case = "case_1"
    case_path = case + "/"
    directory_path = "Texts/"
    while True:
        directory = input("Auto (Enter) or Manual (manual): ")
        if directory == "manual" or directory == "":
            break
        else:
            print("\n*** Please press enter or type 'manual' then type. ***")
    if directory:
        directory_path = directory_path + directory + "/"
    path = global_path + case_path + directory_path
    while True:
        try:
            net_txt = open(path + "netlist.txt")
            break
        except:
            print("\n*** Cannot found the file: " + path + "netlist.txt ***")
            case = input("Please input the correct case: ")
            sub_path = path.split("/")
            path = ""
            for i in range(len(sub_path)):
                if i == 2:
                    path += case + "/"
                elif i != len(sub_path) - 1:
                    path += sub_path[i] + "/"
            print()
    layer = input("Layer (ex: Cu-1): ")
    if len(layer) == 0:
        layer = "Cu-1"
    while True:
        try:
            cline_txt = open(path + "clines_" + layer + ".txt")
            pad_via_txt = open(path + "pad_via_" + layer + ".txt")
            break
        except:
            print(
                "\n*** Cannot found the file: "
                + path
                + "clines/pad_via_"
                + layer
                + ".txt ***"
            )
            layer = input("Please input the correct layer: ")
    while True:
        pad_diameter = input("Pad Diameter (ex: 110): ")
        if len(pad_diameter) == 0:
            pad_diameter = "110"
        try:
            pad_diameter = int(pad_diameter)
            break
        except:
            print("\n*** Please input the correct pad size. ***")
            continue
    pt_space = 0
    tt_space = 0
    # while True:
    #     pt_space = input("Pad-Trace Space (ex: 20): ")
    #     if len(pt_space) == 0:
    #         pt_space = '20'
    #     try:
    #         pt_space = int(pt_space)
    #         break
    #     except:
    #         print("\n*** Please input the correct pad-trace space. ***")
    #         continue
    # while True:
    #     tt_space = input("Trace-Trace Space (ex: 20): ")
    #     if len(tt_space) == 0:
    #         tt_space = '20'
    #     try:
    #         tt_space = int(tt_space)
    #         break
    #     except:
    #         print("\n*** Please input the correct trace-trace space. ***")
    #         continue
    while True:
        crossing_range = input("Net crossing range (ex: 15): ")
        if len(crossing_range) == 0:
            crossing_range = "15"
        try:
            crossing_range = int(crossing_range)
            break
        except:
            print("\n*** Please input the correct crossing range. ***")
            continue
    while True:
        pooling_size = input("Pooling size (ex: 200): ")
        if len(pooling_size) == 0:
            pooling_size = "200"
        try:
            pooling_size = int(pooling_size)
            break
        except:
            print("\n*** Please input the correct pooling size. ***")
            continue
    while True:
        allowed_in_block = input(
            "How many nets allowed in a block (can be float, ex: 2): "
        )
        if len(allowed_in_block) == 0:
            allowed_in_block = "2"
        try:
            allowed_in_block = float(allowed_in_block)
            break
        except:
            print("\n*** Please input the correct information. ***")
            continue
    while True:
        opt_flow = input("Which detour optimization flow (1: original, 2: dt): ")
        if len(opt_flow) == 0:
            opt_flow = "1"
        opt_flow = int(opt_flow)
        if opt_flow in [1, 2]:
            break
        else:
            print("\n*** Please input the correct information. ***")
    print("[Input Parameters]")
    print("------------------")
    print("Case Name:", case)
    if not directory:
        print("Directory: Auto")
    else:
        print("Directory: Manual")
    print("Layer:", layer)
    print("Pad Diameter:", pad_diameter, "um")
    # print("Pad-Trace Space:", pt_space, "um")
    # print("Trace-Track Space:", tt_space, "um")
    print("Crossing Range:", crossing_range, "nets")
    print("Pooling Size:", pooling_size)
    print("Dense Parameter (allowed_in_block):", allowed_in_block, "nets")
    print("Detour Optimization Flow:", opt_flow)
    print("------------------")

    total_start = time.time()
    print()
    print("===== Start Refinement =====")
    print("[Input]")
    print("-------")

    net_info, pad_via_info, chip_edge, max_width, grid_size = Parser.Input(
        net_txt, cline_txt, pad_via_txt, layer
    )
    coordinates =  []
    interactive_tool(net_info, pad_via_info, pad_diameter, coordinates)
    print(coordinates)
    if (len(coordinates) != 0):
        net_info=findPartialNets(net_info, coordinates)
    print("===== Finish Selection =====")
    grid_size = math.ceil(grid_size / pooling_size) * pooling_size
    NET = ChangeType.netinfo2NET(net_info)
    prepare_image(net_info, case, pad_diameter)
    pic(net_info, pad_via_info, 1, case, pad_diameter, "input")
    # prepare_dense_image(net_info, pad_via_info, case, pad_diameter)
    # f = open('example.txt', 'w')
    # for i in range(5):
    #     print('name', net_info[i][0], file=f)
    #     print('pin', 'x', net_info[i][1][0], 'y', net_info[i][1][1], file=f)
    #     print('via', 'x', net_info[i][2][0], 'y', net_info[i][2][1], file=f)
    #     print('segments', net_info[i][3], file=f)
    #     print('direction', net_info[i][4], file=f)
    #     print('NET', NET[net_info[i][0]], file=f)
    #     print('', file=f)
    # f.close()
    # return

    print("\n[Find Net-Order]")
    print("----------------")
    netorder = order.get_netorder(
        NET, net_info
    )  # counterclockwise: UP -> LEFT -> DOWN -> RIGHT
    # print(len(netorder), ':')
    # for ls in netorder:
    #     print('\t', len(ls))
    # print(netorder)
    # return
    if len(netorder) > 1:
        print(netorder)

    print("\n[Detour Optimization]")
    print("---------------------")

    # num_dense = [0, 0, 0]
    # print("Heatmap (before) ...")
    # grid, Direct, divide = Grid.create(net_info, grid_size)
    # dense, Direct_pooling, block_size = Grid.pooling(grid, Direct, pooling_size, "dense")
    # if (divide > 1):
    #     d = divide - 1
    # else:
    #     d = divide

    # # print("detect dense ...")
    # dense_blocks = Detection.detect_dense(dense, pooling_size, block_size, allowed_in_block*d)
    # num_dense[0] = len(dense_blocks)

    # max_v = 0
    # for i, j in dense_blocks:
    #     max_v += dense[i][j]

    # max_v /= len(dense_blocks)

    # plt.figure()
    # plt.axis('off')
    # sns.heatmap(dense, square=True, cmap='rainbow', vmax=max_v, vmin=0)
    # plt.savefig(f'../data/{case}/{opt_flow}_heatmap_original.png')
    # plt.close()

    # num_seg = [0, 0, 0]
    # net_leng = [0., 0., 0.]
    # time_record = 0.
    # for net in net_info:
    #     num_seg[0] += len(net[3])
    #     for seg in net[3]:
    #         leng = (abs(seg[1]-seg[0])**2+abs(seg[3]-seg[2])**2)**(1/2)
    #         net_leng[0] += leng

    start_time = time.time()
    if opt_flow == 1:  # original
        # global detour opt
        route.set_net(
            NET,
            netorder,
            pad_via_info,
            pad_diameter,
            max_width,
            pt_space,
            tt_space,
            crossing_range,
            chip_edge,
        )
        route.route_final1(case=case)
        # return
        # NET = route.return_()
        # net_info = ChangeType.NET2netinfo(NET, net_info)
        # pic(net_info, 0, case, pad_diameter)

        # local detour opt
        route.route_final1(GL=0)
        NET = route.return_()
        net_info = ChangeType.NET2netinfo(NET, net_info)
        pic(net_info, pad_via_info, 1, case, pad_diameter, f"{opt_flow}_detour")
    else:
        if opt_flow == 2:  # dt_min
            dt1.set_net(
                NET,
                netorder,
                pad_via_info,
                pad_diameter,
                max_width,
                pt_space,
                tt_space,
                crossing_range,
                chip_edge,
            )
            dt1.route_final1(case=case)

            dt1.route_final1(GL=0)
            NET = dt1.return_()
            net_info = ChangeType.NET2netinfo(NET, net_info)
            pic(net_info, pad_via_info, 1, case, pad_diameter, f"{opt_flow}_detour")
        elif opt_flow == 3:  # dt_all
            dt2.set_net(
                NET,
                netorder,
                pad_via_info,
                pad_diameter,
                max_width,
                pt_space,
                tt_space,
                crossing_range,
                chip_edge,
            )
            dt2.route_final1(GL=0)
            NET = dt2.return_()
            net_info = ChangeType.NET2netinfo(NET, net_info)
            pic(net_info, pad_via_info, 1, case, pad_diameter, f"{opt_flow}_detour")
    end_time = time.time()

    # for net in net_info:
    #     num_seg[1] += len(net[3])
    #     for seg in net[3]:
    #         leng = (abs(seg[1]-seg[0])**2+abs(seg[3]-seg[2])**2)**(1/2)
    #         net_leng[1] += leng
    # time_record = end_time - start_time

    # print("Heatmap (middle) ...")
    # grid, Direct, divide = Grid.create(net_info, grid_size)
    # dense, Direct_pooling, block_size = Grid.pooling(grid, Direct, pooling_size, "dense")
    # if (divide > 1):
    #     d = divide - 1
    # else:
    #     d = divide

    # # print("detect dense ...")
    # dense_blocks = Detection.detect_dense(dense, pooling_size, block_size, allowed_in_block*d)
    # num_dense[1] = len(dense_blocks)

    # plt.figure()
    # plt.axis('off')
    # sns.heatmap(dense, square=True, cmap='rainbow', vmax=max_v, vmin=0)
    # plt.savefig(f'../data/{case}/{opt_flow}_heatmap_detour.png')
    # plt.close()

    print()
    print("[Area Optimization]")
    print("-------------------")
    print()
    print(" -[Global Translation]")
    print(" =====================")
    move.set_global(
        NET,
        netorder,
        pad_via_info,
        pad_diameter,
        max_width,
        pt_space,
        tt_space,
        crossing_range,
        chip_edge,
    )
    move.movereset()
    for i in range(7):
        move.cut()
    # net_info = ChangeType.NET2netinfo(NET, net_info)
    # pic(net_info, pad_via_info, 1, case, pad_diameter, "cut")

    # for transRound in range(1):
    for transRound in range(12):

        print("Round " + str(transRound + 1) + ": ", end="")
        start = time.time()

        # normal netorder (counterclockwise: UP->LEFT->DOWN->RIGHT)
        # w/ counterclockwise flipping
        for sub_netorder in netorder:
            for i in range(len(sub_netorder)):
                # if sub_netorder[i] == 'TX_CC1' or sub_netorder[i] == 'TX_CC2':
                #     print(sub_netorder[i])
                move.move_all(sub_netorder, i, 1)
                # if sub_netorder[i] == 'TX_CC1' or sub_netorder[i] == 'TX_CC2':
                #     print(sub_netorder[i])
        # net_info = ChangeType.NET2netinfo(NET, net_info)
        # pic(net_info, pad_via_info, 1, case, pad_diameter, "in_g_area_"+str(transRound)+"_1")
        for i in range(7):
            move.cut()
        # net_info = ChangeType.NET2netinfo(NET, net_info)
        # pic(net_info, pad_via_info, 1, case, pad_diameter, "in_g_area_"+str(transRound)+"_2")

        # reversed netorder (clockwise: RIGHT->DOWN->LEFT->UP)
        # w/ clockwise flipping
        for sub_netorder in netorder:
            for i in range(len(sub_netorder) - 1, 0, -1):
                # if sub_netorder[i] == 'TX_CC1' or sub_netorder[i] == 'TX_CC2':
                #     print(sub_netorder[i])
                move.move_all(sub_netorder, i, 0)
                # if sub_netorder[i] == 'TX_CC1' or sub_netorder[i] == 'TX_CC2':
                #     print(sub_netorder[i])
        # net_info = ChangeType.NET2netinfo(NET, net_info)
        # pic(net_info, pad_via_info, 1, case, pad_diameter, "in_g_area_"+str(transRound)+"_3")
        for i in range(7):
            move.cut()
        # net_info = ChangeType.NET2netinfo(NET, net_info)
        # pic(net_info, pad_via_info, 1, case, pad_diameter, "in_g_area_"+str(transRound)+"_4")

        # enlarge the distince between nets
        move.move50()
        net_info = ChangeType.NET2netinfo(NET, net_info)

        end = time.time()
        print(format(end - start))
        # pic(net_info, pad_via_info, 1, case, pad_diameter, "g_area_"+str(transRound))
    end_time = time.time()

    # blank detection
    net_info = ChangeType.NET2netinfo(NET, net_info)
    # print(grid_size)
    # print(pooling_size)
    blank_result = Detection.detect_area(
        net_info, grid_size, pooling_size, allowed_in_block
    )

    print()
    print(" -[Local Translation]")
    print(" ====================")
    start = time.time()
    for sub_netorder in netorder:
        move.local(sub_netorder, blank_result)
    end = time.time()
    print("Finish local net translation: ", format(end - start))
    net_info = ChangeType.NET2netinfo(NET, net_info)
    # pic(net_info, pad_via_info, 1, case, pad_diameter, "l_area")

    print()
    print("[Bends Reduction]")
    print("-----------------")
    start = time.time()
    for sub_netorder in netorder:
        for i in range(len(sub_netorder)):
            move.move_flip(sub_netorder, i, 1)
        for i in range(len(sub_netorder) - 1, 0, -1):
            move.move_flip(sub_netorder, i, 0)
        for i in range(len(sub_netorder)):
            move.move_flip(sub_netorder, i, 1)
        for i in range(len(sub_netorder) - 1, 0, -1):
            move.move_flip(sub_netorder, i, 0)
    for i in range(7):
        move.cut()
    end = time.time()
    print("Finish bends reduction: ", format(end - start))
    net_info = ChangeType.NET2netinfo(NET, net_info)
    # pic(net_info, pad_via_info, 1, case, pad_diameter, "bend")

    # for net in net_info:
    #     num_seg[2] += len(net[3])
    #     for seg in net[3]:
    #         leng = (abs(seg[1]-seg[0])**2+abs(seg[3]-seg[2])**2)**(1/2)
    #         net_leng[2] += leng

    # print("Heatmap (result) ...")
    # grid, Direct, divide = Grid.create(net_info, grid_size)
    # dense, Direct_pooling, block_size = Grid.pooling(grid, Direct, pooling_size, "dense")
    # if (divide > 1):
    #     d = divide - 1
    # else:
    #     d = divide

    # # print("detect dense ...")
    # dense_blocks = Detection.detect_dense(dense, pooling_size, block_size, allowed_in_block*d)
    # num_dense[2] = len(dense_blocks)

    # plt.figure()
    # plt.axis('off')
    # sns.heatmap(dense, square=True, cmap='rainbow', vmax=max_v, vmin=0)
    # plt.savefig(f'../data/{case}/{opt_flow}_heatmap_result.png')
    # plt.close()

    print()
    print("[Output]")
    print("--------")
    Parser.Output(path, layer, net_info, pad_diameter)

    print()
    print("[Finish Refinement]")
    print("-------------------")
    total_end = time.time()
    print("Total Time: ", format(total_end - total_start))
    pic(net_info, pad_via_info, 1, case, pad_diameter, f"{opt_flow}_refined")

    # f = open(f'../data/{case}/{opt_flow}.log', 'w')
    # print('original number of segments:', num_seg[0], '| length: ', net_leng[0], file=f)
    # print('original number of dense area', num_dense[0], file=f)
    # print('after detour opt.:', num_seg[1], '| length: ', net_leng[1], file=f)
    # print('                  ', num_dense[1], file=f)
    # print('final result', num_seg[2], '| length: ', net_leng[2], file=f)
    # print('            ', num_dense[2], file=f)
    # print('timing in detour:', time_record, file=f)
    # print('total time:', total_end - total_start, file=f)
    # f.close()
    # print('original number of segments:', num_seg[0], '| length: ', net_leng[0])
    # print('original number of dense area', num_dense[0])
    # print('after detour opt.:', num_seg[1], '| length: ', net_leng[1])
    # print('                  ', num_dense[1])
    # print('final result', num_seg[2], '| length: ', net_leng[2])
    # print('            ', num_dense[2])
    # print('timing in detour:', time_record)
    # print('total time:', total_end - total_start)


# #     end = time.time()
# #     print(format(end-start))
#     print("[")
#     for i in range(len(netorder)):
#         print("[")
#         print("[","'",netorder[i],"'","]",",",
#               "[",NET[netorder[ i ]][0][0],",",NET[netorder[ i ]][0][2],"],",
#               "[",NET[netorder[ i ]][-1][1],",",NET[netorder[ i ]][-1][3],"],",
#               NET[netorder[ i ]]
#              ,"]",",")
#     print("]")

if __name__ == "__main__":
    main()
