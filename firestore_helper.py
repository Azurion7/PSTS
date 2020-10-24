from firebase_admin import credentials, firestore
import firebase_admin


class FirestoreHelper:
    def __init__(self, creds_path):
        self.__cred = credentials.Certificate(creds_path)
        self.default_app = firebase_admin.initialize_app(self.__cred)
        self.db = firestore.client()

    def update_status(self, name, status):
        self.db.collection('ParkingStatus').document(name).update(
            {'status': status}
        )

    def update_counts(self, name, n_occupied, n_empty):
        self.db.collection('ParkingLot').document(name).update(
            {
                'occupied': n_occupied,
                'empty': n_empty
             }
        )
