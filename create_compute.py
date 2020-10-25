from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "cluster-compute"

try:
    # Try to get running cluster with specified name
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, using it.')
except ComputeTargetException:
    # Exception occurs when there is no cluster created with the specified name. Create new cluster in this case
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', max_nodes=4)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
        
        
