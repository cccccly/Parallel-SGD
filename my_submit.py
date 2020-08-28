if __name__ == '__main__':

    from network import NodeAssignment, Request
    nodes = NodeAssignment()
    nodes.add(0, '127.0.0.1')
    net = Request()

    from roles import Coordinator
    with net.request(nodes) as req:
        master = Coordinator(req)

        from executor.myExe import myExecutor
        master.submit_job(myExecutor)
        master.join()
