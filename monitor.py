class Monitor:
    def __init__(self, config):
        self.config = config
        self.open_positions = {}

    def add_position(self, contract):
        self.open_positions[contract['id']] = contract

    async def check_open_positions(self, api):
        for contract_id, contract in list(self.open_positions.items()):
            try:
                updated_contract = await api.proposal_open_contract(contract_id)
                if updated_contract['status'] == 'closed':
                    del self.open_positions[contract_id]
                elif self.should_apply_trailing_stop(updated_contract):
                    await self.apply_trailing_stop(api, updated_contract)
            except Exception as e:
                print(f"Error checking position {contract_id}: {e}")

    def should_apply_trailing_stop(self, contract):
        if 'current_spot' not in contract or 'entry_spot' not in contract:
            return False
        profit_pips = (contract['current_spot'] - contract['entry_spot']) * 10000
        return profit_pips > self.config.TRAILING_STOP_PIPS

    async def apply_trailing_stop(self, api, contract):
        new_stop_loss = contract['current_spot'] - (self.config.TRAILING_STOP_PIPS / 10000)
        try:
            await api.sell({
                "contract_id": contract['id'],
                "price": new_stop_loss
            })
        except Exception as e:
            print(f"Error applying trailing stop to {contract['id']}: {e}")

